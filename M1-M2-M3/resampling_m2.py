import json
import numpy as np
from resampling_m1 import MIStructure, intersect_ray_sphere


def extract_matrices_for_bbox(data) -> list[np.ndarray]:
    matrices = []
    if isinstance(data, list):
        for item in data:
            matrices.extend(extract_matrices_for_bbox(item))
    elif isinstance(data, dict):
        if "matrix_world" in data:
            matrices.append(np.array(data["matrix_world"], dtype=float))
        if "target" in data:
            matrices.extend(extract_matrices_for_bbox(data["target"]))
        if "tool" in data:
            matrices.extend(extract_matrices_for_bbox(data["tool"]))
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


def interval_intersection(a_list: list[tuple[float, float]], b_list: list[tuple[float, float]]) -> list[tuple[float, float]]:
    res = []
    for a_in, a_out in a_list:
        for b_in, b_out in b_list:
            start = max(a_in, b_in)
            end = min(a_out, b_out)
            if start < end - 1e-9:
                res.append((start, end))
    return interval_union(res)


def interval_difference(a_list: list[tuple[float, float]], b_list: list[tuple[float, float]]) -> list[tuple[float, float]]:
    current = list(a_list)
    for b_in, b_out in b_list:
        next_step = []
        for a_in, a_out in current:
            if b_out <= a_in + 1e-9 or b_in >= a_out - 1e-9:
                next_step.append((a_in, a_out))
            else:
                if a_in < b_in - 1e-9:
                    next_step.append((a_in, b_in))
                if b_out < a_out - 1e-9:
                    next_step.append((b_out, a_out))
        current = next_step
    return current


def evaluate_node_for_ray(
    node,
    orig_world: np.ndarray,
    dir_world: np.ndarray,
    inv_cache: dict,
    ignore_boolean_ops: bool = False
) -> list[tuple[float, float]]:
    if isinstance(node, list):
        all_intervals = []
        for item in node:
            all_intervals.extend(evaluate_node_for_ray(item, orig_world, dir_world, inv_cache, ignore_boolean_ops))
        return interval_union(all_intervals)

    if isinstance(node, dict):
        if "operation" in node and "target" in node and "tool" in node:
            target_iv = evaluate_node_for_ray(node["target"], orig_world, dir_world, inv_cache, ignore_boolean_ops)
            tool_iv = evaluate_node_for_ray(node["tool"], orig_world, dir_world, inv_cache, ignore_boolean_ops)

            if ignore_boolean_ops:
                return interval_union(target_iv + tool_iv)

            op = node["operation"].upper()
            if op == "UNION":
                return interval_union(target_iv + tool_iv)
            elif op == "INTERSECTION":
                return interval_intersection(target_iv, tool_iv)
            elif op == "DIFFERENCE":
                return interval_difference(target_iv, tool_iv)
            else:
                raise ValueError(f"Operazione non supportata: {op}")

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
    ignore_boolean_ops: bool = False
) -> MIStructure:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    matrices = extract_matrices_for_bbox(data)
    bbox_min, bbox_max = compute_scene_bbox(matrices)
    step = np.linalg.norm(bbox_max - bbox_min) * step_percentage

    grid_min = np.floor(bbox_min / step).astype(int) - 1
    grid_max = np.ceil(bbox_max / step).astype(int) + 1
    box = (
        int(grid_min[0]), int(grid_min[1]), int(grid_min[2]),
        int(grid_max[0]), int(grid_max[1]), int(grid_max[2])
    )

    mi = MIStructure(step=step, box=box)
    min_x, min_y, min_z, max_x, max_y, max_z = box

    # Dimensioni esatte del volume richieste da mi_convert: max - min
    nx = max_x - min_x
    ny = max_y - min_y
    nz = max_z - min_z
    inv_cache = {}

    # PLANE 0 (X): row = Z (0..nz-1), col = Y (0..ny-1)
    dir_x = np.array([1.0, 0.0, 0.0, 0.0])
    for row in range(nz):
        for col in range(ny):
            orig = np.array([0.0, (col + min_y) * step, (row + min_z) * step, 1.0])
            intervals = evaluate_node_for_ray(data, orig, dir_x, inv_cache, ignore_boolean_ops)
            if intervals:
                mi.add_line(0, row, col, [val / step for pair in intervals for val in pair])

    # PLANE 1 (Y): row = X (0..nx-1), col = Z (0..nz-1)
    dir_y = np.array([0.0, 1.0, 0.0, 0.0])
    for row in range(nx):
        for col in range(nz):
            orig = np.array([(row + min_x) * step, 0.0, (col + min_z) * step, 1.0])
            intervals = evaluate_node_for_ray(data, orig, dir_y, inv_cache, ignore_boolean_ops)
            if intervals:
                mi.add_line(1, row, col, [val / step for pair in intervals for val in pair])
    # PLANE 2 (Z): row = Y (0..ny-1), col = X (0..nx-1)
    dir_z = np.array([0.0, 0.0, 1.0, 0.0])
    for row in range(ny):
        for col in range(nx):
            orig = np.array([(col + min_x) * step, (row + min_y) * step, 0.0, 1.0])
            intervals = evaluate_node_for_ray(data, orig, dir_z, inv_cache, ignore_boolean_ops)
            if intervals:
                mi.add_line(2, row, col, [val / step for pair in intervals for val in pair])

    mi.save_mit(output_mit_path)
    return mi
