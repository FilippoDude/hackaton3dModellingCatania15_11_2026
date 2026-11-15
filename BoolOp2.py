
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_JSON = BASE_DIR / "operations.json"
INPUT_DIR = BASE_DIR / "inputObj"
INTERMEDIATE_DIR = BASE_DIR / "intermediate"
OUTPUT_DIR = BASE_DIR / "outputObj"

DEFAULT_STEP = 0.346410155

AXES = {
    0: (1, 2),  # X rays: fixed Y, Z
    1: (0, 2),  # Y rays: fixed X, Z
    2: (0, 1),  # Z rays: fixed X, Y
}


# ============================================================
# JSON
# ============================================================

def load_operations():

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        operations = json.load(f)

    if not isinstance(operations, list):
        raise ValueError("JSON must contain a list of operations.")

    return operations


# ============================================================
# OBJ LOADING
# ============================================================

def load_mesh(obj_filename, matrix_world):

    path = INPUT_DIR / obj_filename

    if not path.exists():
        raise FileNotFoundError(f"Missing OBJ: {path}")

    mesh = trimesh.load(
        str(path),
        force="mesh",
        process=True
    )

    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"Invalid mesh: {obj_filename}")

    matrix = np.asarray(matrix_world, dtype=float)

    if matrix.shape != (4, 4):
        raise ValueError("matrix_world must be 4x4.")

    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix_world contains invalid values.")

    if abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
        raise ValueError("matrix_world is singular.")

    mesh.apply_transform(matrix)

    print(f"Loaded: {obj_filename}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    print(f"Watertight: {mesh.is_watertight}")

    return mesh


# ============================================================
# GRID GENERATION
# ============================================================

def generate_grid(meshes, step):

    bounds = np.array([
        mesh.bounds for mesh in meshes
    ])

    minimum = bounds[:, 0, :].min(axis=0)
    maximum = bounds[:, 1, :].max(axis=0)

    # Grid is anchored at world coordinate (0, 0, 0).
    grid_min = np.floor(minimum / step).astype(int) - 1
    grid_max = np.ceil(maximum / step).astype(int) + 1

    axes = [
        np.arange(grid_min[i], grid_max[i] + 1)
        * step
        for i in range(3)
    ]

    print("\nGrid configuration:")
    print("Minimum:", grid_min)
    print("Maximum:", grid_max)
    print("Step:", step)

    return grid_min, grid_max, axes


# ============================================================
# RAY CASTING
# ============================================================

def cast_axis_rays(mesh, axis, coordinates, step):

    """
    Cast parallel rays along one axis.

    Returns:
        Dictionary:
            (row, column) -> [entry, exit, entry, exit...]
    """

    fixed_axes = AXES[axis]

    first_values = coordinates[fixed_axes[0]]
    second_values = coordinates[fixed_axes[1]]

    first_grid = np.arange(len(first_values))
    second_grid = np.arange(len(second_values))

    pairs = [
        (int(i), int(j))
        for i in first_grid
        for j in second_grid
    ]

    origins = np.zeros((len(pairs), 3), dtype=float)

    direction = np.zeros(3)
    direction[axis] = 1.0

    bounds = mesh.bounds

    margin = max(
        np.ptp(bounds[:, axis]) * 0.1,
        step * 2.0
    )

    ray_start = bounds[0, axis] - margin

    origins[:, axis] = ray_start

    for ray_id, (row, column) in enumerate(pairs):

        origins[ray_id, fixed_axes[0]] = first_values[row]
        origins[ray_id, fixed_axes[1]] = second_values[column]

    directions = np.tile(direction, (len(origins), 1))

    # Perform all parallel rays in one call.
    locations, ray_ids, _ = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=True
    )

    # Calculate intersection positions along the ray.
    distances = locations[:, axis]

    hits = {}

    for ray_id, distance in zip(ray_ids, distances):

        hits.setdefault(int(ray_id), []).append(float(distance))

    result = {}

    tolerance = max(step * 1e-6, 1e-9)

    for ray_id, (row, column) in enumerate(pairs):

        intersections = sorted(hits.get(ray_id, []))

        # Remove duplicate hits caused by shared triangle edges.
        unique = []

        for value in intersections:

            if not unique or abs(value - unique[-1]) > tolerance:
                unique.append(value)

        # Pair intersections using even-odd parity.
        if len(unique) % 2 != 0:

            print(
                f"WARNING: Odd number of intersections "
                f"on axis {axis}, line {row} {column}. "
                "Discarding unmatched final intersection."
            )

            unique = unique[:-1]

        if unique:

            result[(row, column)] = unique

    return result


# ============================================================
# OBJ -> MIT
# ============================================================

def generate_mit(mesh, filename, grid_min, grid_max, axes, step):

    planes = {}

    for axis in range(3):

        print(f"\nGenerating plane {axis}...")

        planes[axis] = cast_axis_rays(
            mesh,
            axis,
            axes,
            step
        )

        print(
            f"Plane {axis}: "
            f"{len(planes[axis])} non-empty lines"
        )

    write_mit(
        filename,
        planes,
        grid_min,
        grid_max,
        step
    )

    return planes


# ============================================================
# MIT WRITER
# ============================================================

def write_mit(filename, planes, grid_min, grid_max, step):

    with open(filename, "w", encoding="utf-8") as f:

        f.write("# MI_FORMAT_VERSION 1.0\n\n")

        f.write(f"STEP {step:.9f}\n\n")

        f.write(
            "# Integer grid box: min_x min_y min_z "
            "max_x max_y max_z\n\n"
        )

        f.write(
            "BOX "
            + " ".join(map(str, grid_min))
            + " "
            + " ".join(map(str, grid_max))
            + "\n\n"
        )

        for axis in range(3):

            f.write(f"PLANE {axis}\n\n")

            for (row, column), intersections in sorted(
                planes[axis].items()
            ):

                values = " ".join(
                    f"{value:.9f}"
                    for value in intersections
                )

                f.write(
                    f"LINE {row} {column} {values}\n"
                )

            f.write("\nEND_PLANE\n\n")

    print(f"MIT exported: {filename}")


# ============================================================
# INTERVAL UNION
# ============================================================

def merge_intervals(intervals, tolerance=1e-8):

    """
    Merge overlapping or touching intervals.

    Input:
        [(a, b), (c, d), ...]

    Output:
        [(minimum, maximum), ...]
    """

    if not intervals:
        return []

    intervals = sorted(
        (min(a, b), max(a, b))
        for a, b in intervals
    )

    merged = [list(intervals[0])]

    for start, end in intervals[1:]:

        previous = merged[-1]

        if start <= previous[1] + tolerance:

            previous[1] = max(
                previous[1],
                end
            )

        else:

            merged.append([start, end])

    return [
        (start, end)
        for start, end in merged
    ]


def union_planes(target_planes, tool_planes):

    result = {}

    for axis in range(3):

        target = target_planes[axis]
        tool = tool_planes[axis]

        all_keys = set(target.keys()) | set(tool.keys())

        result[axis] = {}

        for key in all_keys:

            intervals = []

            for values in (
                target.get(key, []),
                tool.get(key, [])
            ):

                for i in range(0, len(values) - 1, 2):

                    intervals.append(
                        (values[i], values[i + 1])
                    )

            merged = merge_intervals(intervals)

            if merged:

                result[axis][key] = [
                    value
                    for interval in merged
                    for value in interval
                ]

    return result


# ============================================================
# MIT -> VOLUME
# ============================================================

def mit_to_volume(planes, axes):

    """
    Reconstruct occupancy using the X-direction
    intersection intervals.

    Volume indexing: [X, Y, Z].
    """

    xs, ys, zs = axes

    volume = np.zeros(
        (len(xs), len(ys), len(zs)),
        dtype=bool
    )

    x_plane = planes[0]

    for (row, column), intersections in x_plane.items():

        y_index = row
        z_index = column

        if y_index >= len(ys) or z_index >= len(zs):
            continue

        for i in range(0, len(intersections) - 1, 2):

            start = intersections[i]
            end = intersections[i + 1]

            inside = (
                (xs >= start) &
                (xs <= end)
            )

            volume[:, y_index, z_index] |= inside

    return volume


# ============================================================
# MARCHING CUBES
# ============================================================

def volume_to_obj(volume, grid_min, step, output_path):

    if not np.any(volume):
        raise RuntimeError("Union volume is empty.")

    # Marching Cubes extracts the 0.5 isosurface.
    vertices, faces, normals, values = marching_cubes(
        volume.astype(np.float32),
        level=0.5,
        spacing=(step, step, step)
    )

    # Convert grid-relative vertices to world coordinates.
    vertices += grid_min * step

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=True
    )

    mesh.export(str(output_path))

    print("\nMarching Cubes completed.")
    print("Vertices:", len(mesh.vertices))
    print("Faces:", len(mesh.faces))
    print("Watertight:", mesh.is_watertight)
    print("OBJ exported:", output_path)

    return mesh


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="MIT-based Boolean Union"
    )

    parser.add_argument(
        "--step",
        type=float,
        default=DEFAULT_STEP
    )

    args = parser.parse_args()

    if args.step <= 0:
        raise ValueError("Grid step must be positive.")

    INTERMEDIATE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("================================")
    print("MIT BOOLEAN UNION")
    print("================================")

    operations = load_operations()

    if not operations:
        raise ValueError("No operations found.")

    final_mesh = None

    for operation in operations:

        if operation.get("operation", "").upper() != "UNION":
            raise ValueError(
                f"Unsupported operation: {operation.get('operation')}"
            )

        operation_id = operation["id"]

        print(f"\nOperation ID: {operation_id}")

        target = operation["target"]
        tool = operation["tool"]

        # Load transformed input geometry.
        target_mesh = load_mesh(
            target["obj_snapshot"],
            target["matrix_world"]
        )

        tool_mesh = load_mesh(
            tool["obj_snapshot"],
            tool["matrix_world"]
        )

        # Generate one common grid for both meshes.
        grid_min, grid_max, axes = generate_grid(
            [target_mesh, tool_mesh],
            args.step
        )

        # Generate individual MIT files.
        target_mit = INTERMEDIATE_DIR / (
            f"{operation_id}_target.mit"
        )

        tool_mit = INTERMEDIATE_DIR / (
            f"{operation_id}_tool.mit"
        )

        target_planes = generate_mit(
            target_mesh,
            target_mit,
            grid_min,
            grid_max,
            axes,
            args.step
        )

        tool_planes = generate_mit(
            tool_mesh,
            tool_mit,
            grid_min,
            grid_max,
            axes,
            args.step
        )

        # Perform interval-based Boolean Union.
        print("\nPerforming MIT Boolean Union...")

        union_planes_data = union_planes(
            target_planes,
            tool_planes
        )

        union_mit = OUTPUT_DIR / "union.mit"

        write_mit(
            union_mit,
            union_planes_data,
            grid_min,
            grid_max,
            args.step
        )

        # Reconstruct the volumetric representation.
        print("\nReconstructing union volume...")

        volume = mit_to_volume(
            union_planes_data,
            axes
        )

        # Extract final surface.
        final_obj = OUTPUT_DIR / "union.obj"

        final_mesh = volume_to_obj(
            volume,
            grid_min,
            args.step,
            final_obj
        )

    print("\n================================")
    print("BOOLEAN UNION COMPLETED")
    print("================================")

    print("MIT:", OUTPUT_DIR / "union.mit")
    print("OBJ:", OUTPUT_DIR / "union.obj")


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
