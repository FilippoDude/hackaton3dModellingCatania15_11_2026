"""M4 - rasterizzazione di una mesh di triangoli (cubo, passo 10%).
M5 - strutture sottili (passo 10%, 5%, 1%, 0.2%)."""
import os
import subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # percorsi relativi a questa cartella

from mesh import Mesh
from mi import rasterize_mesh, save_mit

# (file di input, nome di output, passi in percentuale della diagonale del bounding box)
JOBS = [
    ("./inputs/cube.obj", "m4_cube", [10]),
    ("./inputs/concave.obj", "m5_concave", [10, 5, 1, 0.2]),
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

    for output in outputs:
        subprocess.run([command, output + ".mit", output + ".obj"])
