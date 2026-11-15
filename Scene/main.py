"""Scena di presentazione: struttura MI della mesh della scena a piu' passi."""
import os
import subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # percorsi relativi a questa cartella

from mesh import Mesh
from mi import rasterize_mesh, save_mit

# (file di input, nome di output, passi in percentuale della diagonale del bounding box)
JOBS = [
    ("./inputs/scene.obj", "scene", [20, 10, 5, 3, 2, 1, 0.5, 0.2]),
]

if __name__ == "__main__":
    os.makedirs("./outputs", exist_ok=True)

    outputs = []
    for input_path, name, percents in JOBS:
        mesh = Mesh.from_obj(input_path)
        for percent in percents:
            # Con un solo passo il nome resta <name>, altrimenti <name>_<passo>
            output = f"./outputs/{name}" if len(percents) == 1 else f"./outputs/{name}_{percent}"
            step, box, planes = rasterize_mesh(mesh, percent)
            save_mit(output + ".mit", step, box, planes)
            print(f"Generato {output}.mit | STEP: {step:.9f} | BOX: {box}")
            outputs.append(output)

    default = os.path.normpath("../tools/mi_convert.exe")
    command = input(f"Dimmi la path del comando mi_convert (default: {default}): ").strip() or default

    failed = []
    for output in outputs:
        result = subprocess.run([command, output + ".mit", output + ".obj"])
        if result.returncode != 0 or not os.path.exists(output + ".obj"):
            failed.append(output)
    if failed:
        print("mi_convert non ha generato: " + ", ".join(f"{o}.obj" for o in failed))
        raise SystemExit(1)
