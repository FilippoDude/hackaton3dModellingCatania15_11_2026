"""Operazioni booleane (UNION, INTERSECTION, DIFFERENCE) tra strutture MI.

Le strutture MI degli oggetti vengono create sulla stessa griglia (stesso step e box),
cosi' ogni linea del target corrisponde alla stessa linea del tool e l'operazione si fa
linea per linea sugli intervalli dentro/fuori, come nelle slide.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "M4-M5"))

from mesh import Mesh  # noqa: E402
from mi import enforce_consistency, grid_box, one_per_edge, rasterize_mesh  # noqa: E402

OPERATIONS = {
    "UNION": lambda a, b: a or b,
    "INTERSECTION": lambda a, b: a and b,
    "DIFFERENCE": lambda a, b: a and not b,
}


def boolean_line(a, b, operation):
    """Combina le intersezioni di due linee: si percorre la linea, ogni intersezione
    cambia lo stato dentro/fuori del suo oggetto e si emette un'intersezione ogni volta
    che cambia lo stato del risultato."""
    inside = OPERATIONS[operation]
    state = [False, False]
    result = []
    for t, obj in sorted([(t, 0) for t in a] + [(t, 1) for t in b]):
        before = inside(*state)
        state[obj] = not state[obj]
        if inside(*state) != before:
            result.append(t)
    return one_per_edge(result)  # toglie i doppioni (es. contatto in un punto)


def boolean_planes(target, tool, operation, box):
    """Applica l'operazione a tutte le linee dei tre piani."""
    if operation not in OPERATIONS:
        raise ValueError(f"Operazione '{operation}' non supportata: usa {', '.join(OPERATIONS)}")
    planes = {}
    for axis in range(3):
        planes[axis] = {}
        for key in set(target[axis]) | set(tool[axis]):
            values = boolean_line(target[axis].get(key, []), tool[axis].get(key, []), operation)
            if values:
                planes[axis][key] = values
    enforce_consistency(box, planes)
    return planes


def load_object(obj, folder):
    """Mesh <name>.obj con matrix_world applicata (obj_snapshot va ignorato)."""
    matrix = np.array(obj["matrix_world"], dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{obj['name']}: matrix_world deve essere una matrice 4x4")
    mesh = Mesh.from_obj(os.path.join(folder, obj["name"] + ".obj"))
    mesh.vertices = mesh.vertices @ matrix[:3, :3].T + matrix[:3, 3]
    return mesh


def run_operations(json_path, step):
    """Esegue in ordine le operazioni del JSON e restituisce (step, box, planes) del
    target dell'ultima operazione. Ogni oggetto e' tenuto per nome: la prima volta si
    rasterizza la sua mesh, poi si riusa il volume (gia' modificato dalle operazioni)
    finche' la trasformazione resta la stessa."""
    folder = os.path.dirname(json_path)
    with open(json_path, encoding="utf-8") as f:
        operations = json.load(f)

    # Griglia comune: deve contenere tutti gli oggetti in tutte le loro posizioni
    meshes = {}
    for op in operations:
        for role in ("target", "tool"):
            obj = op[role]
            key = (obj["name"], json.dumps(obj["matrix_world"]))
            if key not in meshes:
                meshes[key] = load_object(obj, folder)
    box = grid_box(np.vstack([m.vertices for m in meshes.values()]), step)

    volumes = {}  # nome -> (matrix_world, planes)
    for op in operations:
        planes = {}
        for role in ("target", "tool"):
            obj = op[role]
            matrix = json.dumps(obj["matrix_world"])
            if obj["name"] in volumes and volumes[obj["name"]][0] == matrix:
                planes[role] = volumes[obj["name"]][1]
            else:
                # Prima volta (o trasformazione cambiata): rasterizza la mesh originale.
                # Per M9 servira' estrarre la mesh dal volume se l'oggetto era gia' stato modificato.
                planes[role] = rasterize_mesh(meshes[(obj["name"], matrix)], step=step, box=box)[2]
        result = boolean_planes(planes["target"], planes["tool"], op["operation"].upper(), box)
        volumes[op["target"]["name"]] = (json.dumps(op["target"]["matrix_world"]), result)
        print(f"Operazione {op.get('id')}: {op['target']['name']} {op['operation']} {op['tool']['name']}")

    return step, box, volumes[operations[-1]["target"]["name"]][1]
