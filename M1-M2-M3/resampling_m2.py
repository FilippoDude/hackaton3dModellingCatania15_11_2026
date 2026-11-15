import json
import numpy as np
from resampling_m1 import MIStructure, intersect_ray_sphere


def extract_matrices_for_bbox(data) -> list[np.ndarray]:
    """Estrae tutte le matrix_world presenti nel JSON per calcolare il Bounding Box globale."""
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
    """Calcola il bounding box globale trasformando gli 8 vertici del cubo [-1, 1]^3."""
    unit_corners = np.array([
        [x, y, z, 1.0]
        for x in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    ]).T  # shape (4, 8)

    all_pts = []
    for M in matrices:
        transformed = (M @ unit_corners)[:3, :].T
        all_pts.append(transformed)

    all_pts = np.vstack(all_pts)
    return all_pts.min(axis=0), all_pts.max(axis=0)


# ==========================================
# ALGEBRA BOOLEANA 1D SU INTERVALLI [t_in, t_out]
# ==========================================

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
    """Calcola A - B per liste di intervalli disgiunti."""
    current = list(a_list)
    for b_in, b_out in b_list:
        next_step = []
        for a_in, a_out in current:
            # Nessuna sovrapposizione
            if b_out <= a_in + 1e-9 or b_in >= a_out - 1e-9:
                next_step.append((a_in, a_out))
            else:
                # Porzione sinistra superstite
                if a_in < b_in - 1e-9:
                    next_step.append((a_in, b_in))
                # Porzione destra superstite
                if b_out < a_out - 1e-9:
                    next_step.append((b_out, a_out))
        current = next_step
    return current


# ==========================================
# VALUTAZIONE RICORSIVA DEL RAGGIO SUL JSON
# ==========================================

def evaluate_node_for_ray(node, orig_world: np.ndarray, dir_world: np.ndarray, inv_cache: dict) -> list[tuple[float, float]]:
    """Valuta un nodo del JSON (oggetto singolo, lista o operazione booleana) lungo un raggio."""
    if isinstance(node, list):
        # Se è una lista di elementi (es. [obj] o [op1, op2]), unisce i risultati
        all_intervals = []
        for item in node:
            all_intervals.extend(evaluate_node_for_ray(item, orig_world, dir_world, inv_cache))
        return interval_union(all_intervals)

    if isinstance(node, dict):
        # Caso 1: Nodo operazione booleana (target, tool, operation)
        if "operation" in node and "target" in node and "tool" in node:
            op = node["operation"].upper()
            target_iv = evaluate_node_for_ray(node["target"], orig_world, dir_world, inv_cache)
            tool_iv = evaluate_node_for_ray(node["tool"], orig_world, dir_world, inv_cache)

            if op == "UNION":
                return interval_union(target_iv + tool_iv)
            elif op == "INTERSECTION":
                return interval_intersection(target_iv, tool_iv)
            elif op == "DIFFERENCE":
                return interval_difference(target_iv, tool_iv)
            else:
                raise ValueError(f"Operazione booleana non supportata: {op}")

        # Caso 2: Primitiva sfera/ellissoide con matrix_world
        if "matrix_world" in node:
            node_id = id(node)
            if node_id not in inv_cache:
                M = np.array(node["matrix_world"], dtype=float)
                inv_cache[node_id] = np.linalg.inv(M)
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
    step_percentage: float = 0.10
) -> MIStructure:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    matrices = extract_matrices_for_bbox(data)
    bbox_min, bbox_max = compute_scene_bbox(matrices)
    diagonal = np.linalg.norm(bbox_max - bbox_min)
    step = diagonal * step_percentage

    # Griglia allineata sullo 0
    grid_min = np.floor(bbox_min / step).astype(int) - 1
    grid_max = np.ceil(bbox_max / step).astype(int) + 1
    box = (
        int(grid_min[0]), int(grid_min[1]), int(grid_min[2]),
        int(grid_max[0]), int(grid_max[1]), int(grid_max[2])
    )

    mi = MIStructure(step=step, box=box)
    nx = box[3] - box[0] + 1
    ny = box[4] - box[1] + 1
    nz = box[5] - box[2] + 1

    inv_cache = {}

    for axis in (0, 1, 2):
        if axis == 0:
            dir_world = np.array([1.0, 0.0, 0.0, 0.0])
            r_max, c_max = ny, nz
        elif axis == 1:
            dir_world = np.array([0.0, 1.0, 0.0, 0.0])
            r_max, c_max = nx, nz
        else:
            dir_world = np.array([0.0, 0.0, 1.0, 0.0])
            r_max, c_max = nx, ny

        for row in range(r_max):
            for col in range(c_max):
                if axis == 0:
                    orig_world = np.array([0.0, (row + box[1]) * step, (col + box[2]) * step, 1.0])
                elif axis == 1:
                    orig_world = np.array([(row + box[0]) * step, 0.0, (col + box[2]) * step, 1.0])
                else:
                    orig_world = np.array([(row + box[0]) * step, (col + box[1]) * step, 0.0, 1.0])

                intervals = evaluate_node_for_ray(data, orig_world, dir_world, inv_cache)
                if intervals:
                    flat_t = []
                    for t_in, t_out in intervals:
                        flat_t.extend([t_in / step, t_out / step])
                    mi.add_line(axis, row, col, flat_t)

    mi.save_mit(output_mit_path)
    return mi
