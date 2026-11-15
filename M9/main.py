"""M9 - logo Hackmore: sequenza di operazioni booleane da operations.json (step 1.2).
Usa rasterizzazione e .mit di M4-M5 e booleane di M6-M7-M8.

Quando un oggetto gia' modificato viene spostato (la Sphere scavata dall'operazione 20 e
poi usata come tool in due posizioni) bisogna estrarre la mesh dal suo volume:
  - "mi_convert": mesh primale con `mi_convert -m`, come fa mi_process (default)
  - "python":     mesh duale di mi.mi_to_mesh, senza eseguibili esterni
"""
import os
import subprocess
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # percorsi relativi a questa cartella
sys.path.insert(0, "../M6-M7-M8")

from boolean import run_operations  # noqa: E402
from mesh import Mesh  # noqa: E402
from mi import mi_to_mesh, save_mit  # noqa: E402

STEP = 1.2  # passo assoluto, lo stesso usato per la validazione (mi_process -s 1.2)
JSON = "./inputs/operations.json"
OUTPUT = "./outputs/m9_hackmore"
EXTRACTION = sys.argv[1] if len(sys.argv) > 1 else "mi_convert"


def extract_with_mi_convert(command):
    def extract(step, box, planes):
        save_mit("./outputs/extracted.mit", step, box, planes)
        subprocess.run([command, "-m", "./outputs/extracted.mit", "./outputs/extracted.obj"], check=True)
        return Mesh.from_obj("./outputs/extracted.obj")
    return extract


def extract_with_python(step, box, planes):
    mesh = mi_to_mesh(step, box, planes)
    triangles = [t for q in mesh.faces for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]
    return Mesh(mesh.vertices, triangles)


if __name__ == "__main__":
    os.makedirs("./outputs", exist_ok=True)

    default = os.path.normpath("../tools/mi_convert.exe")
    command = input(f"Dimmi la path del comando mi_convert (default: {default}): ").strip() or default
    extract = extract_with_mi_convert(command) if EXTRACTION == "mi_convert" else extract_with_python

    step, box, planes = run_operations(JSON, STEP, extract)
    save_mit(OUTPUT + ".mit", step, box, planes)
    print(f"Generato {OUTPUT}.mit | STEP: {step:.9f} | BOX: {box}")

    result = subprocess.run([command, OUTPUT + ".mit", OUTPUT + ".obj"])
    if result.returncode != 0 or not os.path.exists(OUTPUT + ".obj"):
        print(f"mi_convert non ha generato {OUTPUT}.obj")
        raise SystemExit(1)
