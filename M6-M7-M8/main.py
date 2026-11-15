"""M6 - unione, M7 - intersezione, M8 - differenza tra Cube e Cylinder (step 0.05).
Usa rasterizzazione e formato .mit di M4-M5."""
import os
import subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # percorsi relativi a questa cartella

from boolean import run_operations
from mi import save_mit

STEP = 0.05  # passo assoluto, lo stesso usato per le validazioni (mi_process -s 0.05)

# (file JSON con le operazioni, nome di output)
JOBS = [
    ("./inputs/union.json", "m6_union"),
    ("./inputs/intersection.json", "m7_intersection"),
    ("./inputs/difference.json", "m8_difference"),
]

if __name__ == "__main__":
    os.makedirs("./outputs", exist_ok=True)

    outputs = []
    for json_path, name in JOBS:
        output = f"./outputs/{name}"
        step, box, planes = run_operations(json_path, STEP)
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
