import json
import numpy as np
from resampling_m1 import MIStructure, intersect_ray_sphere


def extract_matrices_for_bbox(data) -> list[np.ndarray]:
    """Estrae tutte le matrici di trasformazione presenti nel JSON."""
    matrices = []
    if isinstance(data, list):
        for item in data:
            matrices.extend(extract_matrices_for_bbox(item))
    elif isinstance(data, dict):
        if "matrix_world" in data:
            matrices.append(np.array(data["matrix_world"], dtype=float))
        # Supporto di sicurezza per eventuali nodi annidati senza operazioni
        for key in ["target", "tool", "objects", "spheres"]:
            if key in data:
                matrices.extend(extract_matrices_for_bbox(data[key]))
    return matrices


def compute_scene_bbox(matrices: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    unit_corners = np.array([
        [x, y, z, 1.0]
        for x in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    ]).T

    all_pts = [(M @ unit_corners)[:3, :].T for M in matrices]
    all_pts = np.vstack(all_pts)
    return all_pts.min(axis=0), all_pts.max(axis=0)


def interval_union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_iv[0]]
    for cur_in, cur_out in sorted_iv[1:]:
        prev_in, prev_out = merged[-1]
        if cur_in <= prev_out + 1e-9:
            merged[-1] = (prev_in, max(prev_out, cur_out))
        else:
            merged.append((cur_in, cur_out))
    return merged


def collect_intervals_for_ray(node, orig_world: np.ndarray, dir_world: np.ndarray, inv_cache: dict) -> list[tuple[float, float]]:
    """Raccoglie e unisce gli intervalli di tutte le sfere/ellissoidi incontrate dal raggio."""
    if isinstance(node, list):
        all_ivs = []
        for item in node:
            all_ivs.extend(collect_intervals_for_ray(item, orig_world, dir_world, inv_cache))
        return interval_union(all_ivs)

    if isinstance(node, dict):
        # Se è un nodo contenente target/tool (struttura log di Blender), scansiona entrambi
        if "target" in node or "tool" in node:
            sub_ivs = []
            if "target" in node:
                sub_ivs.extend(collect_intervals_for_ray(node["target"], orig_world, dir_world, inv_cache))
            if "tool" in node:
                sub_ivs.extend(collect_intervals_for_ray(node["tool"], orig_world, dir_world, inv_cache))
            return interval_union(sub_ivs)

        # Se è una primitiva con matrix_world
        if "matrix_world" in node:
            node_id = id(node)
            if node_id not in inv_cache:
                inv_cache[node_id] = np.linalg.inv(np.array(node["matrix_world"], dtype=float))
            M_inv = inv_cache[node_id]

            orig_local = (M_inv @ orig_world)[:3]
            dir_local = (M_inv @ dir_world)[:3]
            hits = intersect_ray_sphere(orig_local, dir_local, np.zeros(3), 1.0)
            if len(hits) == 2:
                return [(hits[0], hits[1])]
            return []

    return []


def rasterize_primitives_json(
    json_path: str,
    output_mit_path: str,
    step_percentage: float = 0.10,
    fixed_step: float = None
) -> MIStructure:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    matrices = extract_matrices_for_bbox(data)
    bbox_min, bbox_max = compute_scene_bbox(matrices)
    
    # Se vuoi forzare lo step esatto a 0.1 come nel file validato, puoi passarlo, altrimenti usa la percentuale
    step = fixed_step if fixed_step is not None else (np.linalg.norm(bbox_max - bbox_min) * step_percentage)

    grid_min = np.floor(bbox_min / step).astype(int) - 1
    grid_max = np.ceil(bbox_max / step).astype(int) + 1
    box = (
        int(grid_min[0]), int(grid_min[1]), int(grid_min[2]),
        int(grid_max[0]), int(grid_max[1]), int(grid_max[2])
    )

    mi = MIStructure(step=step, box=box)
    min_x, min_y, min_z, max_x, max_y, max_z = box
    nx = max_x - min_x
    ny = max_y - min_y
    nz = max_z - min_z
    inv_cache = {}

    # PLANE 0 (X): row = Z, col = Y
    dir_x = np.array([1.0, 0.0, 0.0, 0.0])
    for row in range(nz):
        for col in range(ny):
            orig = np.array([0.0, (col + min_y) * step, (row + min_z) * step, 1.0])
            intervals = collect_intervals_for_ray(data, orig, dir_x, inv_cache)
            if intervals:
                mi.add_line(0, row, col, [val / step for pair in intervals for val in pair])

    # PLANE 1 (Y): row = X, col = Z
    dir_y = np.array([0.0, 1.0, 0.0, 0.0])
    for row in range(nx):
        for col in range(nz):
            orig = np.array([(row + min_x) * step, 0.0, (col + min_z) * step, 1.0])
            intervals = collect_intervals_for_ray(data, orig, dir_y, inv_cache)
            if intervals:
                mi.add_line(1, row, col, [val / step for pair in intervals for val in pair])

    # PLANE 2 (Z): row = Y, col = X
    dir_z = np.array([0.0, 0.0, 1.0, 0.0])
    for row in range(ny):
        for col in range(nx):
            orig = np.array([(col + min_x) * step, (row + min_y) * step, 0.0, 1.0])
            intervals = collect_intervals_for_ray(data, orig, dir_z, inv_cache)
            if intervals:
                mi.add_line(2, row, col, [val / step for pair in intervals for val in pair])

    mi.save_mit(output_mit_path)
    return mi
