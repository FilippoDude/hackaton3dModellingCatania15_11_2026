import argparse
import json
import sys
import shutil
from pathlib import Path

import numpy as np
import trimesh

from skimage.measure import marching_cubes
from scipy.ndimage import label


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

JSON_FILE = BASE_DIR / "operations.json"

INPUT_DIR = BASE_DIR / "inputObj"
INTERMEDIATE_DIR = BASE_DIR / "intermediate"
OUTPUT_DIR = BASE_DIR / "outputObj"

DEFAULT_STEP = 0.2

SUPPORTED_OPERATIONS = {
    "UNION",
    "DIFFERENCE",
    "INTERSECTION"
}

AXIS_FIXED = {
    0: (1, 2),
    1: (0, 2),
    2: (0, 1)
}

# Numero minimo di assi concordi per considerare pieno un voxel
MIN_VALID_AXES = 2

RAY_TOLERANCE_FACTOR = 1e-5


# ============================================================
# LOAD OPERATIONS
# ============================================================

def load_operations():

    if not JSON_FILE.exists():
        raise FileNotFoundError(JSON_FILE)

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        operations = json.load(f)

    if not isinstance(operations, list) or not operations:
        raise ValueError("operations.json must contain a non-empty list.")

    return operations


# ============================================================
# LOAD MESH
# ============================================================

def load_mesh(obj_filename, matrix_world):

    obj_path = INPUT_DIR / obj_filename

    if not obj_path.exists():
        raise FileNotFoundError(obj_path)

    mesh = trimesh.load(
        str(obj_path),
        force="mesh",
        process=True
    )

    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"Invalid mesh: {obj_filename}")

    matrix = np.asarray(matrix_world, dtype=float)

    if matrix.shape != (4, 4):
        raise ValueError("matrix_world must be 4x4.")

    if not np.all(np.isfinite(matrix)):
        raise ValueError("Invalid transformation matrix.")

    if abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
        raise ValueError("Singular transformation matrix.")

    mesh.apply_transform(matrix)

    print(f"\nLoaded mesh: {obj_filename}")
    print("Vertices:", len(mesh.vertices))
    print("Faces:", len(mesh.faces))
    print("Watertight:", mesh.is_watertight)
    print("Is volume:", mesh.is_volume)
    print("Bounds:")
    print(mesh.bounds)

    edges = np.sort(mesh.edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = np.count_nonzero(counts == 1)
    print("Boundary edges:", boundary_edges)

    trimesh.repair.fix_normals(mesh)

    # Ripara eventuali buchi SOLO se la mesh non è già chiusa (watertight)
    if not mesh.is_watertight and boundary_edges > 0:
        try:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
        except Exception as error:
            print("Hole repair warning:", error)

    print("Watertight after repair:", mesh.is_watertight)

    return mesh


# ============================================================
# GENERATE GRID
# ============================================================

def generate_grid(meshes, step):

    bounds = np.array([mesh.bounds for mesh in meshes])

    world_min = bounds[:, 0, :].min(axis=0)
    world_max = bounds[:, 1, :].max(axis=0)

    # Grid coordinates: integer cell indices.
    grid_min = np.floor(world_min / step).astype(np.int64) - 2
    grid_max = np.ceil(world_max / step).astype(np.int64) + 2

    dimensions = grid_max - grid_min

    axes = [
        (grid_min[i] + np.arange(dimensions[i]) + 0.5) * step
        for i in range(3)
    ]

    print("\n========== GRID ==========")
    print("World minimum:", world_min)
    print("World maximum:", world_max)
    print("Grid minimum:", grid_min)
    print("Grid maximum:", grid_max)
    print("Dimensions:", dimensions)
    print("Step:", step)

    return grid_min, grid_max, axes


# ============================================================
# RAY CASTING
# ============================================================

def cast_axis(mesh, axis, coordinates, step):

    fixed_axes = AXIS_FIXED[axis]

    first_values = coordinates[fixed_axes[0]]
    second_values = coordinates[fixed_axes[1]]

    dimensions = tuple(len(a) for a in coordinates)

    occupancy = np.zeros(dimensions, dtype=bool)
    valid = np.zeros(dimensions, dtype=bool)

    pairs = [
        (row, column)
        for row in range(len(first_values))
        for column in range(len(second_values))
    ]

    origins = np.zeros((len(pairs), 3), dtype=float)

    direction = np.zeros(3, dtype=float)
    direction[axis] = 1.0

    ray_margin = max(
        np.ptp(mesh.bounds[:, axis]) + step * 10,
        step * 10
    )
    ray_start = mesh.bounds[0, axis] - ray_margin

    origins[:, axis] = ray_start

    for ray_id, (row, column) in enumerate(pairs):
        origins[ray_id, fixed_axes[0]] = first_values[row]
        origins[ray_id, fixed_axes[1]] = second_values[column]

    directions = np.tile(direction, (len(origins), 1))

    locations, ray_ids, _ = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=True
    )

    hits = {}
    for ray_id, location in zip(ray_ids, locations):
        hits.setdefault(int(ray_id), []).append(float(location[axis]))

    intervals = {}
    odd_rays = 0
    valid_rays = 0

    tolerance = max(step * RAY_TOLERANCE_FACTOR, 1e-8)
    axis_coordinates = coordinates[axis]

    for ray_id, (row, column) in enumerate(pairs):

        intersections = sorted(hits.get(ray_id, []))

        # Remove duplicate triangle-edge intersections.
        unique = []
        for position in intersections:
            if not unique or abs(position - unique[-1]) > tolerance:
                unique.append(position)

        if len(unique) % 2 != 0:
            odd_rays += 1
            continue

        valid_rays += 1

        slices = [slice(None)] * 3
        slices[fixed_axes[0]] = row
        slices[fixed_axes[1]] = column

        valid[tuple(slices)] = True

        ray_intervals = []
        for i in range(0, len(unique), 2):
            start = unique[i]
            end = unique[i + 1]
            if end <= start:
                continue
            ray_intervals.append((start, end))

        if ray_intervals:
            intervals[(row, column)] = [
                value
                for interval in ray_intervals
                for value in interval
            ]

        inside = np.zeros(len(axis_coordinates), dtype=bool)
        for start, end in ray_intervals:
            inside |= (axis_coordinates >= start) & (axis_coordinates <= end)

        occupancy[tuple(slices)] = inside

    print(
        f"Axis {axis}: valid rays={valid_rays}, "
        f"ambiguous rays={odd_rays}, "
        f"lines={len(intervals)}"
    )

    return intervals, occupancy, valid


# ============================================================
# GENERATE THREE-AXIS VOLUME
# ============================================================

def generate_mesh_volume(mesh, axes, step):

    axis_volumes = []
    axis_validity = []
    planes = {}

    for axis in range(3):
        print(f"\nProcessing axis {axis}...")
        intervals, occupancy, valid = cast_axis(mesh, axis, axes, step)
        planes[axis] = intervals
        axis_volumes.append(occupancy)
        axis_validity.append(valid)

    votes = np.zeros(axis_volumes[0].shape, dtype=np.uint8)
    valid_votes = np.zeros_like(votes)

    for occupancy, valid in zip(axis_volumes, axis_validity):
        votes += (occupancy & valid).astype(np.uint8)
        valid_votes += valid.astype(np.uint8)

    # Voto di maggioranza: evita buchi o gradini errati se un singolo raggio è tangente
    volume = (
        (valid_votes >= MIN_VALID_AXES)
        & (votes * 2 >= valid_votes)
    )

    # Clip the volume to the actual mesh bounding box.
    for axis in range(3):
        inside_bounds = (
            (axes[axis] >= mesh.bounds[0, axis] - step * 0.5)
            & (axes[axis] <= mesh.bounds[1, axis] + step * 0.5)
        )
        shape = [1, 1, 1]
        shape[axis] = len(axes[axis])
        volume &= inside_bounds.reshape(shape)

    print("\n========== VOLUME DEBUG ==========")
    for axis in range(3):
        print(f"Axis {axis} occupied:", np.count_nonzero(axis_volumes[axis]))

    print("Valid classifications:", np.count_nonzero(valid_votes))
    print("Final occupied voxels:", np.count_nonzero(volume))

    return planes, volume


# ============================================================
# WRITE MIT
# ============================================================

def write_mit(filename, planes, grid_min, grid_max, step):

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# MI_FORMAT_VERSION 1.0\n\n")
        f.write(f"STEP {step:.9f}\n\n")
        f.write(
            "BOX "
            + " ".join(map(str, grid_min))
            + " "
            + " ".join(map(str, grid_max))
            + "\n\n"
        )

        for axis in range(3):
            f.write(f"PLANE {axis}\n\n")
            for (row, column), values in sorted(planes[axis].items()):
                formatted = " ".join(f"{value:.9f}" for value in values)
                f.write(f"LINE {row} {column} {formatted}\n")
            f.write("\nEND_PLANE\n\n")

    print("MIT saved:", filename)


# ============================================================
# INTERVAL OPERATIONS
# ============================================================

def merge_intervals(intervals, tolerance=1e-8):

    if not intervals:
        return []

    intervals = sorted((min(a, b), max(a, b)) for a, b in intervals)
    merged = [list(intervals[0])]

    for start, end in intervals[1:]:
        previous = merged[-1]
        if start <= previous[1] + tolerance:
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])

    return [(a, b) for a, b in merged]


def subtract_intervals(a, b):

    result = []
    for start, end in a:
        remaining = [(start, end)]
        for b_start, b_end in b:
            updated = []
            for r_start, r_end in remaining:
                if b_end <= r_start or b_start >= r_end:
                    updated.append((r_start, r_end))
                else:
                    if b_start > r_start:
                        updated.append((r_start, b_start))
                    if b_end < r_end:
                        updated.append((b_end, r_end))
            remaining = updated
        result.extend(remaining)

    return merge_intervals(result)


def intersect_intervals(a, b):

    result = []
    i = 0
    j = 0
    a = merge_intervals(a)
    b = merge_intervals(b)

    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            result.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1

    return merge_intervals(result)


def boolean_intervals(a, b, operation):

    if operation == "UNION":
        return merge_intervals(a + b)
    if operation == "DIFFERENCE":
        return subtract_intervals(a, b)
    if operation == "INTERSECTION":
        return intersect_intervals(a, b)
    raise ValueError(operation)


# ============================================================
# BOOLEAN MIT PLANES
# ============================================================

def boolean_planes(target_planes, tool_planes, operation):

    result = {}

    for axis in range(3):
        target = target_planes[axis]
        tool = tool_planes[axis]
        keys = set(target) | set(tool)
        result[axis] = {}

        for key in keys:
            a_values = target.get(key, [])
            b_values = tool.get(key, [])

            a = [(a_values[i], a_values[i + 1]) for i in range(0, len(a_values) - 1, 2)]
            b = [(b_values[i], b_values[i + 1]) for i in range(0, len(b_values) - 1, 2)]

            intervals = boolean_intervals(a, b, operation)
            if intervals:
                result[axis][key] = [value for interval in intervals for value in interval]

    return result


# ============================================================
# BOOLEAN 3D VOLUMES
# ============================================================

def boolean_volumes(target, tool, operation):

    if target.shape != tool.shape:
        raise ValueError("Target and tool volume dimensions differ.")

    if operation == "UNION":
        return target | tool
    if operation == "DIFFERENCE":
        return target & ~tool
    if operation == "INTERSECTION":
        return target & tool
    raise ValueError(operation)


# ============================================================
# RECONSTRUCT OBJ
# ============================================================

def volume_to_obj(volume, grid_min, step, output_path, keep_largest_only=False):

    occupied = np.count_nonzero(volume)

    print("\n========== RECONSTRUCTION ==========")
    print("Volume dimensions:", volume.shape)
    print("Occupied voxels:", occupied)

    if occupied == 0:
        raise RuntimeError("Boolean result is empty.")

    # Rimuove solo rumore isolato di 1-2 voxel (non intere parti valide)
    structure = np.ones((3, 3, 3), dtype=int)
    labeled_volume, num_features = label(volume, structure=structure)
    if num_features > 1:
        component_sizes = np.bincount(labeled_volume.ravel())
        component_sizes[0] = 0
        if keep_largest_only:
            largest_component = component_sizes.argmax()
            volume = (labeled_volume == largest_component)
        else:
            # Filtra solo artefatti minuscoli (< 4 voxel) mantenendo eventuali multi-parti reali
            valid_labels = np.where(component_sizes >= 4)[0]
            if len(valid_labels) > 0:
                volume = np.isin(labeled_volume, valid_labels)

    padded = np.pad(
        volume.astype(np.uint8),
        1,
        mode="constant",
        constant_values=0
    )

    vertices, faces, normals, values = marching_cubes(
        padded,
        level=0.5,
        spacing=(step, step, step)
    )

    # Allineamento esatto al centro dei voxel della griglia
    origin = (np.asarray(grid_min, dtype=float) - 0.5) * step
    vertices += origin

    # process=False evita che trimesh alteri o fonda vertici in modo inatteso
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        process=False
    )

    trimesh.repair.fix_normals(mesh)

    mesh.export(str(output_path))

    print("\nOBJ reconstruction completed.")
    print("Vertices:", len(mesh.vertices))
    print("Faces:", len(mesh.faces))
    print("Watertight:", mesh.is_watertight)
    print("Bounds:")
    print(mesh.bounds)
    print("Saved:", output_path)

    return mesh


# ============================================================
# PROCESS OPERATION
# ============================================================

def process_operation(operation, step):

    operation_id = operation.get("id", "unknown")
    operation_type = operation.get("operation", "").upper()

    if operation_type not in SUPPORTED_OPERATIONS:
        raise ValueError(f"Unsupported operation: {operation_type}")

    print("\n================================")
    print("OPERATION ID:", operation_id)
    print("OPERATION:", operation_type)
    print("================================")

    target = operation["target"]
    tool = operation["tool"]

    target_mesh = load_mesh(target["obj_snapshot"], target["matrix_world"])
    tool_mesh = load_mesh(tool["obj_snapshot"], tool["matrix_world"])

    grid_min, grid_max, axes = generate_grid([target_mesh, tool_mesh], step)

    print("\nGenerating TARGET representation...")
    target_planes, target_volume = generate_mesh_volume(target_mesh, axes, step)

    print("\nGenerating TOOL representation...")
    tool_planes, tool_volume = generate_mesh_volume(tool_mesh, axes, step)

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_mit(INTERMEDIATE_DIR / f"{operation_id}_target.mit", target_planes, grid_min, grid_max, step)
    write_mit(INTERMEDIATE_DIR / f"{operation_id}_tool.mit", tool_planes, grid_min, grid_max, step)

    print(f"\nPerforming MIT {operation_type}...")
    result_planes = boolean_planes(target_planes, tool_planes, operation_type)
    result_mit = OUTPUT_DIR / f"{operation_type.lower()}_{operation_id}.mit"
    write_mit(result_mit, result_planes, grid_min, grid_max, step)

    print(f"\nPerforming 3D {operation_type}...")
    print("Target voxels:", np.count_nonzero(target_volume))
    print("Tool voxels:", np.count_nonzero(tool_volume))

    result_volume = boolean_volumes(target_volume, tool_volume, operation_type)
    print("Result voxels:", np.count_nonzero(result_volume))

    obj_path = OUTPUT_DIR / f"{operation_type.lower()}_{operation_id}.obj"
    result_mesh = volume_to_obj(result_volume, grid_min, step, obj_path)

    return result_mesh, result_mit, obj_path


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(description="MIT Boolean Processor")
    parser.add_argument("--step", type=float, default=DEFAULT_STEP)
    args = parser.parse_args()

    if args.step <= 0:
        raise ValueError("Grid step must be positive.")

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("================================")
    print("MIT BOOLEAN PROCESSOR")
    print("================================")

    operations = load_operations()
    print("Loaded", len(operations), "operation(s).")

    results = []
    for operation in operations:
        results.append(process_operation(operation, args.step))

    final_mesh, final_mit, final_obj = results[-1]
    generic_operation = operations[-1]["operation"].lower()

    generic_mit = OUTPUT_DIR / f"{generic_operation}.mit"
    generic_obj = OUTPUT_DIR / f"{generic_operation}.obj"

    final_mesh.export(str(generic_obj))

    if generic_mit != final_mit:
        shutil.copy2(final_mit, generic_mit)

    print("\n================================")
    print("PROCESS COMPLETED")
    print("================================")
    print("Operation:", generic_operation.upper())
    print("MIT:", generic_mit)
    print("OBJ:", generic_obj)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)