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


def matrix_of(obj):
    matrix = np.array(obj["matrix_world"], dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{obj['name']}: matrix_world deve essere una matrice 4x4")
    return matrix


def transformed(mesh, matrix):
    """Copia della mesh con la matrice 4x4 applicata ai vertici."""
    with np.errstate(all="ignore"):  # numpy 2.0 su macOS: falsi avvisi 'divide by zero in matmul'
        return Mesh(mesh.vertices @ matrix[:3, :3].T + matrix[:3, 3], mesh.faces)


class SceneObject:
    """Un oggetto della scena di Blender: la sua mesh in coordinate locali, la
    trasformazione corrente e il volume MI in coordinate mondo."""

    def __init__(self, mesh, matrix):
        self.mesh = mesh          # coordinate locali
        self.matrix = matrix      # matrix_world con cui e' stato calcolato il volume
        self.planes = None
        self.modified = False     # il volume e' cambiato rispetto a self.mesh


def run_operations(json_path, step, extract_mesh=None):
    """Esegue in ordine le operazioni del JSON e restituisce (step, box, planes) del
    target dell'ultima operazione, seguendo la logica Blender -> MI delle slide:
    - la prima volta che un oggetto compare si legge <name>.obj (obj_snapshot va
      ignorato), si applica matrix_world e si crea il volume;
    - se ricompare con la stessa trasformazione si riusa il volume;
    - se la trasformazione e' cambiata si ricalcola il volume: dalla mesh originale se
      l'oggetto non e' stato modificato, altrimenti prima si estrae la mesh dal volume
      con extract_mesh(step, box, planes) -> Mesh (coordinate mondo)."""
    folder = os.path.dirname(json_path)
    with open(json_path, encoding="utf-8") as f:
        operations = json.load(f)

    objects = {}
    for op in operations:
        for role in ("target", "tool"):
            obj = op[role]
            if obj["name"] not in objects:
                mesh = Mesh.from_obj(os.path.join(folder, obj["name"] + ".obj"))
                objects[obj["name"]] = SceneObject(mesh, matrix_of(obj))

    # Griglia comune: deve contenere ogni oggetto in ogni posizione in cui compare
    corners = []
    for op in operations:
        for role in ("target", "tool"):
            obj = op[role]
            corners.append(transformed(objects[obj["name"]].mesh, matrix_of(obj)).vertices)
    box = grid_box(np.vstack(corners), step)

    for op in operations:
        planes = {}
        for role in ("target", "tool"):
            obj = objects[op[role]["name"]]
            matrix = matrix_of(op[role])
            if obj.planes is not None and np.array_equal(obj.matrix, matrix):
                planes[role] = obj.planes
                continue
            if obj.modified:
                if extract_mesh is None:
                    raise ValueError(f"{op[role]['name']} e' stato modificato e poi spostato: "
                                     "serve extract_mesh per estrarne la mesh dal volume")
                # Mesh del volume attuale, riportata in coordinate locali
                world = extract_mesh(step, box, obj.planes)
                obj.mesh = transformed(world, np.linalg.inv(obj.matrix))
                obj.modified = False
                print(f"  {op[role]['name']}: mesh estratta dal volume ({len(world.faces)} facce)")
            obj.matrix = matrix
            obj.planes = rasterize_mesh(transformed(obj.mesh, matrix), step=step, box=box)[2]
            planes[role] = obj.planes

        target = objects[op["target"]["name"]]
        target.planes = boolean_planes(planes["target"], planes["tool"], op["operation"].upper(), box)
        target.modified = True
        print(f"Operazione {op.get('id')}: {op['target']['name']} {op['operation']} {op['tool']['name']}")

    return step, box, objects[operations[-1]["target"]["name"]].planes
