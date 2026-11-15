from resampling_m1 import rasterize_sphere_m1
from resampling_m2 import rasterize_primitives_json
import os

if __name__ == "__main__":
    os.makedirs("./outputs", exist_ok=True)

    mi_sphere = rasterize_sphere_m1(
        center=(0.0, 0.0, 0.0),
        radius=1.0,
        step_percentage=0.10,
        output_path="./outputs/m1_sphere.mit"
    )
    print(f"Generato m1_sphere.mit | STEP: {mi_sphere.step:.9f} | BOX: {mi_sphere.box}")

    rasterize_primitives_json("./inputs/oriented_ellipsoid.json", "./outputs/m2_ellipsoid.mit", step_percentage=0.01325)
    print("Generato m2_ellipsoid.mit")

    # ignore_boolean_ops=True mantiene entrambe le sfere nel file di output per M3
    rasterize_primitives_json(
        "./inputs/two_spheres.json",
        "./outputs/m3_two_spheres.mit"
    )
    print("Generato m3_two_spheres.mit")

    command = input("Dimmi la path del comando mi_convert (default: mi_convert): ").strip()
    if command == "":
        command = "mi_convert"

    os.system(f"{command} ./outputs/m1_sphere.mit ./outputs/m1_sphere.obj")
    os.system(f"{command} ./outputs/m2_ellipsoid.mit ./outputs/m2_ellipsoid.obj")
    os.system(f"{command} ./outputs/m3_two_spheres.mit ./outputs/m3_two_spheres.obj")
