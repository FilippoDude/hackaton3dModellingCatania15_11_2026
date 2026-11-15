#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Boolean mesh operations using a Marching Intersections (MI) pipeline.

Input JSON schema (one or more records):
[
  {
    "id": 43,
    "operation": "UNION",  # UNION | INTERSECTION | DIFFERENCE
    "target": {
      "obj_snapshot": "000043_target.obj",
      "matrix_world": [[...], [...], [...], [...]]
    },
    "tool": {
      "obj_snapshot": "000043_tool.obj",
      "matrix_world": [[...], [...], [...], [...]]
    }
  }
]

The output is an ASCII .mit file in the custom MIT 1.0 format documented
in write_mit(). Mesh coordinates are exported in world space.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh
from skimage.measure import marching_cubes


OPERATION_ALIASES = {
    "UNION": "UNION",
    "OR": "UNION",
    "INTERSECTION": "INTERSECTION",
    "INTERSECT": "INTERSECTION",
    "AND": "INTERSECTION",
    "DIFFERENCE": "DIFFERENCE",
    "SUBTRACTION": "DIFFERENCE",
    "SUBTRACT": "DIFFERENCE",
    "MINUS": "DIFFERENCE",
}


def parse_operation(value: Any) -> str:
    """Normalize and validate the operation name from JSON."""
    key = str(value).strip().upper()
    if key not in OPERATION_ALIASES:
        supported = ", ".join(("UNION", "INTERSECTION", "DIFFERENCE"))
        raise ValueError(f"Operazione '{value}' non supportata. Usa: {supported}.")
    return OPERATION_ALIASES[key]


def boolean_value(operation: str, inside_a: bool, inside_b: bool) -> bool:
    if operation == "UNION":
        return inside_a or inside_b
    if operation == "INTERSECTION":
        return inside_a and inside_b
    if operation == "DIFFERENCE":
        return inside_a and not inside_b
    raise ValueError(f"Operazione interna non valida: {operation}")


def resolve_asset_path(
    asset_value: str,
    json_dir: Path,
    assets_dir: Path | None,
) -> Path:
    """Resolve OBJ paths relative to JSON, --assets-dir, or current directory."""
    candidate = Path(asset_value).expanduser()
    if candidate.is_absolute():
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"File OBJ non trovato: {candidate}")

    search_roots = [json_dir]
    if assets_dir is not None:
        search_roots.append(assets_dir.expanduser().resolve())
    search_roots.append(Path.cwd())

    for root in search_roots:
        possible = (root / candidate).resolve()
        if possible.is_file():
            return possible

    searched = ", ".join(str((root / candidate).resolve()) for root in search_roots)
    raise FileNotFoundError(
        f"File OBJ '{asset_value}' non trovato. Percorsi controllati: {searched}"
    )


def mesh_quality(mesh: trimesh.Trimesh) -> dict[str, int | bool]:
    """Return a small diagnostic summary of mesh edge connectivity."""
    if len(mesh.faces) == 0:
        return {
            "watertight": False,
            "winding_consistent": False,
            "boundary_edges": 0,
            "nonmanifold_edges": 0,
        }

    edge_counts = np.bincount(
        np.asarray(mesh.edges_unique_inverse, dtype=np.int64),
        minlength=len(mesh.edges_unique),
    )
    return {
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "boundary_edges": int(np.count_nonzero(edge_counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(edge_counts > 2)),
    }


def cleanup_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Remove common OBJ artefacts and make face orientation consistent."""
    mesh.process(validate=True)
    if len(mesh.faces):
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces):
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh, multibody=True)
    return mesh


def repair_mesh(
    mesh: trimesh.Trimesh,
    object_label: str,
    obj_path: Path,
    repair_mode: str = "auto",
) -> tuple[trimesh.Trimesh, str]:
    """Try conservative Trimesh repairs, then optional MeshFix if requested/available.

    MeshFix may alter geometry. It is therefore only used by the explicit `auto`
    or `meshfix` modes and its result is accepted only when it improves topology.
    """
    if repair_mode not in {"auto", "none", "meshfix"}:
        raise ValueError(f"Modalità di riparazione sconosciuta: {repair_mode}")

    before = mesh_quality(mesh)
    actions: list[str] = []

    if repair_mode == "none":
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError(
                f"La mesh '{object_label}' ({obj_path.name}) non è valida e "
                "--repair none impedisce la riparazione. "
                f"Diagnostica: boundary_edges={before['boundary_edges']}, "
                f"nonmanifold_edges={before['nonmanifold_edges']}, "
                f"watertight={before['watertight']}, "
                f"winding_consistent={before['winding_consistent']}."
            )
        return mesh, "riparazione disabilitata; mesh già valida"

    if repair_mode != "none":
        mesh = cleanup_mesh(mesh)
        after_cleanup = mesh_quality(mesh)
        if after_cleanup != before:
            actions.append("pulizia di vertici/facce duplicati o degeneri")

        if not mesh.is_watertight:
            # Trimesh can close straightforward triangular and quadrilateral holes.
            try:
                trimesh.repair.fill_holes(mesh)
                mesh = cleanup_mesh(mesh)
            except Exception as exc:
                actions.append(f"riempimento fori semplici non riuscito ({exc})")
            else:
                if mesh.is_watertight:
                    actions.append("chiusura automatica dei fori semplici")

        if not mesh.is_watertight and repair_mode in {"auto", "meshfix"}:
            try:
                from pymeshfix import MeshFix
            except ImportError:
                if repair_mode == "meshfix":
                    raise RuntimeError(
                        "Per la riparazione avanzata installa pymeshfix: "
                        "python -m pip install pymeshfix"
                    )
            else:
                try:
                    fixer = MeshFix(
                        np.asarray(mesh.vertices, dtype=np.float64),
                        np.asarray(mesh.faces, dtype=np.int32),
                        verbose=False,
                    )
                    fixer.repair(
                        joincomp=False,
                        remove_smallest_components=False,
                    )
                    candidate = trimesh.Trimesh(
                        vertices=np.asarray(fixer.points, dtype=np.float64),
                        faces=np.asarray(fixer.faces, dtype=np.int64),
                        process=True,
                    )
                    candidate = cleanup_mesh(candidate)
                    candidate_quality = mesh_quality(candidate)
                    current_quality = mesh_quality(mesh)
                    # Accept a closed result, or a strict reduction in connectivity defects.
                    old_defects = int(current_quality["boundary_edges"]) + int(
                        current_quality["nonmanifold_edges"]
                    )
                    new_defects = int(candidate_quality["boundary_edges"]) + int(
                        candidate_quality["nonmanifold_edges"]
                    )
                    if candidate.is_watertight or new_defects < old_defects:
                        mesh = candidate
                        actions.append("riparazione avanzata MeshFix")
                except Exception as exc:
                    actions.append(f"MeshFix non ha completato la riparazione ({exc})")

    mesh = cleanup_mesh(mesh)
    quality = mesh_quality(mesh)
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        actions_text = "; ".join(actions) if actions else "nessuna riparazione eseguita"
        raise ValueError(
            f"La mesh '{object_label}' ({obj_path.name}) non è ancora un solido valido. "
            f"Diagnostica: boundary_edges={quality['boundary_edges']}, "
            f"nonmanifold_edges={quality['nonmanifold_edges']}, "
            f"watertight={quality['watertight']}, "
            f"winding_consistent={quality['winding_consistent']}. "
            f"Tentativi: {actions_text}. "
            "Prova a installare pymeshfix ('python -m pip install pymeshfix') "
            "oppure ripara la mesh nel software 3D di origine."
        )

    report = ", ".join(actions) if actions else "mesh già valida; nessuna modifica"
    print(
        f"  Riparazione {object_label}: {report}. "
        f"boundary_edges={quality['boundary_edges']}, "
        f"nonmanifold_edges={quality['nonmanifold_edges']}"
    )
    return mesh, report


def load_transformed_mesh(
    object_data: dict[str, Any],
    json_dir: Path,
    assets_dir: Path | None,
    object_label: str,
    repair_mode: str = "auto",
) -> tuple[trimesh.Trimesh, Path, str]:
    """Load an OBJ, bake scene transforms, then apply matrix_world from JSON."""
    if not isinstance(object_data, dict):
        raise ValueError(f"La sezione '{object_label}' deve essere un oggetto JSON.")

    obj_name = object_data.get("obj_snapshot")
    if not obj_name:
        raise ValueError(f"Manca '{object_label}.obj_snapshot' nel JSON.")
    obj_path = resolve_asset_path(str(obj_name), json_dir, assets_dir)

    loaded = trimesh.load(obj_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
        if not geometries:
            raise ValueError(f"Nessuna geometria mesh trovata in {obj_path}.")
        mesh = trimesh.util.concatenate(geometries)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    else:
        raise TypeError(f"Il file {obj_path} non contiene una mesh triangolare.")

    matrix = object_data.get("matrix_world")
    if matrix is not None:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"'{object_label}.matrix_world' deve essere una matrice 4x4; "
                f"ricevuta forma {matrix.shape}."
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"'{object_label}.matrix_world' contiene valori non finiti.")
        mesh.apply_transform(matrix)

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"La mesh '{object_label}' è vuota: {obj_path}.")

    mesh, repair_report = repair_mesh(mesh, object_label, obj_path, repair_mode)
    return mesh, obj_path, repair_report


def unique_sorted_hits(values: Iterable[float], tolerance: float) -> np.ndarray:
    """Sort ray hits and collapse duplicate hits on shared triangle edges."""
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return np.empty(0, dtype=np.float64)

    values.sort()
    groups: list[list[float]] = [[float(values[0])]]
    for value in values[1:]:
        if float(value) - groups[-1][-1] <= tolerance:
            groups[-1].append(float(value))
        else:
            groups.append([float(value)])
    return np.asarray([float(np.mean(group)) for group in groups], dtype=np.float64)


def cast_axis_intersections(
    mesh: trimesh.Trimesh,
    axis: int,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    ray_start: float,
    batch_size: int,
    jitter: float,
    hit_tolerance: float,
    seed: int,
) -> list[np.ndarray]:
    """Cast parallel grid rays and return one sorted hit list per ray.

    The line ordering is (index on first orthogonal axis, index on second
    orthogonal axis). A tiny deterministic jitter avoids rays lying exactly
    on triangle edges/vertices, which otherwise create duplicate/ambiguous hits.
    """
    orthogonal = [dimension for dimension in range(3) if dimension != axis]
    first_axis, second_axis = orthogonal
    first_count = len(grid[first_axis])
    second_count = len(grid[second_axis])
    line_count = first_count * second_count
    hits_by_line: list[list[float]] = [[] for _ in range(line_count)]

    intersector = mesh.ray
    rng = np.random.default_rng(seed)
    direction = np.zeros(3, dtype=np.float64)
    direction[axis] = 1.0

    for batch_begin in range(0, line_count, batch_size):
        batch_end = min(batch_begin + batch_size, line_count)
        flat_indices = np.arange(batch_begin, batch_end, dtype=np.int64)
        first_indices = flat_indices // second_count
        second_indices = flat_indices % second_count
        count = len(flat_indices)

        origins = np.empty((count, 3), dtype=np.float64)
        origins[:, axis] = ray_start
        origins[:, first_axis] = grid[first_axis][first_indices]
        origins[:, second_axis] = grid[second_axis][second_indices]
        if jitter > 0.0:
            origins[:, first_axis] += rng.uniform(-jitter, jitter, size=count)
            origins[:, second_axis] += rng.uniform(-jitter, jitter, size=count)

        directions = np.broadcast_to(direction, (count, 3))
        locations, ray_indices, _ = intersector.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=True,
        )

        if len(locations) == 0:
            continue

        values = locations[:, axis]
        order = np.lexsort((values, ray_indices))
        sorted_ray_indices = ray_indices[order]
        sorted_values = values[order]

        # Append intersections by ray. Deduplication is performed below.
        for local_ray, coordinate in zip(sorted_ray_indices, sorted_values):
            global_ray = batch_begin + int(local_ray)
            hits_by_line[global_ray].append(float(coordinate))

    return [unique_sorted_hits(line_hits, hit_tolerance) for line_hits in hits_by_line]


def combine_intersection_lists(
    hits_a: np.ndarray,
    hits_b: np.ndarray,
    operation: str,
    tolerance: float,
) -> np.ndarray:
    """Apply a Boolean operation to two ordered ray-intersection lists.

    A ray begins outside both solids. Every crossing toggles the corresponding
    in/out state. An output intercept is emitted only when the Boolean result
    changes from outside to inside or vice versa.
    """
    ia = ib = 0
    inside_a = False
    inside_b = False
    output: list[float] = []

    while ia < len(hits_a) or ib < len(hits_b):
        next_a = float(hits_a[ia]) if ia < len(hits_a) else math.inf
        next_b = float(hits_b[ib]) if ib < len(hits_b) else math.inf
        current = min(next_a, next_b)

        group_a: list[float] = []
        group_b: list[float] = []
        while ia < len(hits_a) and float(hits_a[ia]) <= current + tolerance:
            group_a.append(float(hits_a[ia]))
            ia += 1
        while ib < len(hits_b) and float(hits_b[ib]) <= current + tolerance:
            group_b.append(float(hits_b[ib]))
            ib += 1

        before = boolean_value(operation, inside_a, inside_b)
        if len(group_a) % 2:
            inside_a = not inside_a
        if len(group_b) % 2:
            inside_b = not inside_b
        after = boolean_value(operation, inside_a, inside_b)

        if before != after:
            all_group_values = group_a + group_b
            output.append(float(np.mean(all_group_values)))

    return np.asarray(output, dtype=np.float64)


def build_axis_event_map(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    axis: int,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    operation: str,
    batch_size: int,
    jitter: float,
    hit_tolerance: float,
    seed: int,
) -> dict[tuple[int, int], np.ndarray]:
    """Build MI intercept lists for one of the X/Y/Z directions."""
    # The ray must start before the minimum coordinate of the current axis.
    ray_start = float(grid[axis][0] - (grid[axis][1] - grid[axis][0]))
    orthogonal = [dimension for dimension in range(3) if dimension != axis]
    first_count = len(grid[orthogonal[0]])
    second_count = len(grid[orthogonal[1]])

    hits_a = cast_axis_intersections(
        mesh_a, axis, grid, ray_start, batch_size, jitter,
        hit_tolerance, seed=seed + axis * 101,
    )
    hits_b = cast_axis_intersections(
        mesh_b, axis, grid, ray_start, batch_size, jitter,
        hit_tolerance, seed=seed + axis * 101,
    )

    result: dict[tuple[int, int], np.ndarray] = {}
    for first_index in range(first_count):
        for second_index in range(second_count):
            flat_index = first_index * second_count + second_index
            result[(first_index, second_index)] = combine_intersection_lists(
                hits_a[flat_index], hits_b[flat_index], operation, hit_tolerance
            )
    return result


def make_occupancy_grid(
    x_events: dict[tuple[int, int], np.ndarray],
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Classify grid nodes using the Boolean intercept lists along X."""
    nx, ny, nz = (len(axis_grid) for axis_grid in grid)
    occupancy = np.zeros((nx, ny, nz), dtype=np.uint8)
    x_samples = grid[0]

    for iy in range(ny):
        for iz in range(nz):
            events = x_events[(iy, iz)]
            crossings_before_sample = np.searchsorted(events, x_samples, side="right")
            occupancy[:, iy, iz] = (crossings_before_sample & 1).astype(np.uint8)

    return occupancy


def snap_marching_cubes_vertices(
    vertices_index: np.ndarray,
    axis_event_maps: tuple[
        dict[tuple[int, int], np.ndarray],
        dict[tuple[int, int], np.ndarray],
        dict[tuple[int, int], np.ndarray],
    ],
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    step: float,
) -> np.ndarray:
    """Snap each marching-cubes edge vertex to its MI ray intercept.

    Marching Cubes determines cell topology. The corresponding MI list supplies
    the coordinate along the crossed grid edge, rather than retaining the
    linear midpoint/interpolation approximation from the binary occupancy grid.
    """
    snapped = np.asarray(vertices_index, dtype=np.float64).copy()
    integer_tolerance = 1.0e-5
    index_tolerance = 1.0e-4

    for vertex in snapped:
        fractional_parts = np.abs(vertex - np.rint(vertex))
        axis = int(np.argmax(fractional_parts))
        if fractional_parts[axis] <= integer_tolerance:
            continue

        orthogonal = [dimension for dimension in range(3) if dimension != axis]
        line_key = (
            int(np.rint(vertex[orthogonal[0]])),
            int(np.rint(vertex[orthogonal[1]])),
        )
        event_map = axis_event_maps[axis]
        if line_key not in event_map:
            continue

        events = event_map[line_key]
        if len(events) == 0:
            continue

        axis_origin = float(grid[axis][0])
        event_indices = (events - axis_origin) / step
        lower = math.floor(float(vertex[axis])) - index_tolerance
        upper = math.ceil(float(vertex[axis])) + index_tolerance
        candidates = event_indices[(event_indices >= lower) & (event_indices <= upper)]

        if candidates.size == 0:
            continue
        nearest = candidates[int(np.argmin(np.abs(candidates - vertex[axis])))]
        vertex[axis] = float(nearest)

    return snapped


def make_mi_mesh(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    operation: str,
    resolution: int = 96,
    batch_size: int = 2048,
    seed: int = 12345,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Perform a Boolean operation with the Marching Intersections pipeline."""
    if resolution < 8:
        raise ValueError("La risoluzione deve essere almeno 8.")
    if batch_size < 1:
        raise ValueError("batch_size deve essere maggiore di zero.")

    combined_bounds = np.vstack((mesh_a.bounds, mesh_b.bounds))
    minimum = combined_bounds.min(axis=0)
    maximum = combined_bounds.max(axis=0)
    extents = maximum - minimum
    max_extent = float(np.max(extents))
    if not np.isfinite(max_extent) or max_extent <= 0.0:
        raise ValueError("Bounding box dei due oggetti non valida.")

    # Cubic cells: resolution is the number of cells along the longest extent.
    step = max_extent / float(resolution)
    grid_origin = minimum - 2.0 * step
    grid: tuple[np.ndarray, np.ndarray, np.ndarray] = tuple(
        grid_origin[axis]
        + step * np.arange(int(math.ceil((maximum[axis] - grid_origin[axis]) / step)) + 3)
        for axis in range(3)
    )  # type: ignore[assignment]

    coordinate_scale = max(
        max_extent,
        float(np.max(np.abs(combined_bounds))),
        np.finfo(np.float64).tiny,
    )
    hit_tolerance = max(
        step * 1.0e-8, coordinate_scale * np.finfo(np.float64).eps * 16.0
    )
    jitter = max(
        step * 1.0e-6, coordinate_scale * np.finfo(np.float64).eps * 16.0
    )

    print("  Calcolo intercette MI lungo X...", flush=True)
    x_events = build_axis_event_map(
        mesh_a, mesh_b, 0, grid, operation,
        batch_size, jitter, hit_tolerance, seed,
    )
    print("  Calcolo intercette MI lungo Y...", flush=True)
    y_events = build_axis_event_map(
        mesh_a, mesh_b, 1, grid, operation,
        batch_size, jitter, hit_tolerance, seed,
    )
    print("  Calcolo intercette MI lungo Z...", flush=True)
    z_events = build_axis_event_map(
        mesh_a, mesh_b, 2, grid, operation,
        batch_size, jitter, hit_tolerance, seed,
    )

    occupancy = make_occupancy_grid(x_events, grid)
    metadata = {
        "operation": operation,
        "resolution": int(resolution),
        "grid_shape": [int(v) for v in occupancy.shape],
        "grid_spacing": float(step),
        "grid_origin": [float(v) for v in grid_origin],
        "mi_intercepts_x": int(sum(len(v) for v in x_events.values())),
        "mi_intercepts_y": int(sum(len(v) for v in y_events.values())),
        "mi_intercepts_z": int(sum(len(v) for v in z_events.values())),
    }

    if not np.any(occupancy):
        # Valid result for e.g. A - A or intersection of disjoint solids.
        empty = trimesh.Trimesh(
            vertices=np.empty((0, 3), dtype=np.float64),
            faces=np.empty((0, 3), dtype=np.int64),
            process=False,
        )
        return empty, metadata

    if np.all(occupancy):
        raise RuntimeError(
            "Il volume MI risulta interamente interno. Aumenta il padding oppure "
            "controlla che le mesh siano solidi chiusi."
        )

    print("  Ricostruzione superficie con Marching Cubes...", flush=True)
    vertices_index, faces, _, _ = marching_cubes(
        occupancy,
        level=0.5,
        spacing=(1.0, 1.0, 1.0),
        gradient_direction="descent",
        method="lewiner",
        allow_degenerate=False,
    )

    vertices_index = snap_marching_cubes_vertices(
        vertices_index,
        (x_events, y_events, z_events),
        grid,
        step,
    )
    vertices_world = grid_origin + vertices_index * step

    result = trimesh.Trimesh(
        vertices=vertices_world,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    result.process(validate=True)
    result.update_faces(result.unique_faces())
    result.update_faces(result.nondegenerate_faces())
    result.remove_unreferenced_vertices()
    if len(result.faces):
        trimesh.repair.fix_normals(result, multibody=True)

    metadata["vertices"] = int(len(result.vertices))
    metadata["triangles"] = int(len(result.faces))
    metadata["watertight"] = bool(result.is_watertight)
    return result, metadata


def write_mit(
    mesh: trimesh.Trimesh,
    output_path: Path,
    metadata: dict[str, Any],
    operation_id: Any,
    target_name: str,
    tool_name: str,
) -> None:
    """Write the custom ASCII MIT 1.0 mesh format.

    Format specification:
      MIT 1.0
      # comments / metadata
      VERTICES <count>
      v <x> <y> <z>         (one-based vertex records)
      TRIANGLES <count>
      f <i> <j> <k>         (zero-based indices into the vertex array)

    The .mit suffix has no single broadly adopted mesh specification, so this
    script defines a simple, self-describing text format. Coordinates are in
    the same units and world coordinate system as the input OBJ files.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("MIT 1.0\n")
        stream.write("# Marching Intersections mesh output\n")
        stream.write(f"# operation_id: {operation_id}\n")
        stream.write(f"# operation: {metadata['operation']}\n")
        stream.write(f"# target: {target_name}\n")
        stream.write(f"# tool: {tool_name}\n")
        stream.write(f"# target_repair: {metadata.get('target_repair', 'not recorded')}\n")
        stream.write(f"# tool_repair: {metadata.get('tool_repair', 'not recorded')}\n")
        stream.write(f"# grid_shape: {metadata.get('grid_shape', [])}\n")
        stream.write(f"# grid_spacing: {metadata.get('grid_spacing', 0.0):.12g}\n")
        stream.write("# Coordinates use the input OBJ coordinate units, in world space.\n")
        stream.write(f"VERTICES {len(mesh.vertices)}\n")
        for x, y, z in np.asarray(mesh.vertices, dtype=np.float64):
            stream.write(f"v {x:.12g} {y:.12g} {z:.12g}\n")
        stream.write(f"TRIANGLES {len(mesh.faces)}\n")
        for a, b, c in np.asarray(mesh.faces, dtype=np.int64):
            stream.write(f"f {int(a)} {int(b)} {int(c)}\n")


def load_operations(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    if isinstance(data, list):
        operations = data
    elif isinstance(data, dict) and isinstance(data.get("operations"), list):
        operations = data["operations"]
    elif isinstance(data, dict):
        operations = [data]
    else:
        raise ValueError("Il JSON deve contenere una lista di operazioni o un oggetto operazione.")

    if not operations:
        raise ValueError("Il JSON non contiene operazioni.")
    for index, record in enumerate(operations):
        if not isinstance(record, dict):
            raise ValueError(f"L'elemento JSON all'indice {index} non è un oggetto.")
        for required in ("operation", "target", "tool"):
            if required not in record:
                raise ValueError(f"Nell'operazione all'indice {index} manca '{required}'.")
    return operations


def run(args: argparse.Namespace) -> int:
    json_path = args.json_file.expanduser().resolve()
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON non trovato: {json_path}")

    operations = load_operations(json_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = args.assets_dir.expanduser().resolve() if args.assets_dir else None

    for index, record in enumerate(operations, start=1):
        operation = parse_operation(record["operation"])
        operation_id = record.get("id", index)
        print(f"\n[{index}/{len(operations)}] Operazione {operation_id}: {operation}")

        mesh_a, target_path, target_repair = load_transformed_mesh(
            record["target"], json_path.parent, assets_dir, "target", args.repair
        )
        mesh_b, tool_path, tool_repair = load_transformed_mesh(
            record["tool"], json_path.parent, assets_dir, "tool", args.repair
        )
        print(
            f"  Target: {target_path.name} ({len(mesh_a.vertices)} vertici, "
            f"{len(mesh_a.faces)} facce)"
        )
        print(
            f"  Tool:   {tool_path.name} ({len(mesh_b.vertices)} vertici, "
            f"{len(mesh_b.faces)} facce)"
        )

        result, metadata = make_mi_mesh(
            mesh_a,
            mesh_b,
            operation,
            resolution=args.resolution,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        metadata["target_repair"] = target_repair
        metadata["tool_repair"] = tool_repair

        output_path = output_dir / f"operation_{operation_id}_{operation.lower()}.mit"
        write_mit(
            result,
            output_path,
            metadata,
            operation_id,
            str(record["target"].get("name", target_path.name)),
            str(record["tool"].get("name", tool_path.name)),
        )
        print(
            f"  Output: {output_path}\n"
            f"  Mesh risultante: {metadata.get('vertices', 0)} vertici, "
            f"{metadata.get('triangles', 0)} triangoli, "
            f"watertight={metadata.get('watertight', len(result.faces) == 0)}"
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Esegue UNION, INTERSECTION o DIFFERENCE su mesh OBJ usando "
            "Marching Intersections e salva un file ASCII .mit."
        )
    )
    parser.add_argument("json_file", type=Path, help="File JSON con le operazioni")
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=None,
        help="Cartella contenente gli OBJ; se omessa viene cercata anche accanto al JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_mit"),
        help="Cartella di destinazione dei file .mit (default: output_mit)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=96,
        help="Numero di celle sull'asse più lungo (default: 96; più alto = più dettaglio)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Numero di raggi elaborati per batch (default: 2048)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed per il jitter numerico dei raggi (default: 12345)",
    )
    parser.add_argument(
        "--repair",
        choices=("auto", "none", "meshfix"),
        default="auto",
        help=(
            "Riparazione mesh: auto prova pulizia, chiusura fori e MeshFix se installato; "
            "none non modifica la mesh; meshfix richiede pymeshfix (default: auto)"
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        # Give an actionable diagnostic: trimesh's accelerated ray queries need Rtree.
        try:
            import rtree  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Dipendenza mancante: rtree. Installa tutte le dipendenze con "
                "'python -m pip install -r requirements_mi.txt'."
            ) from exc
        return run(args)
    except Exception as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
