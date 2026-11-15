from resampling_m1 import *
from resampling_m2 import *
import os

if __name__ == "__main__":
    mi_sphere = rasterize_sphere_m1(
        center=(0.0, 0.0, 0.0),
        radius=1.0,
        step_percentage=0.10,
        output_path="./outputs/m1_sphere.mit"
    )
    print(f"Generato m1_sphere.mit | STEP: {mi_sphere.step:.9f} | BOX: {mi_sphere.box}")
    rasterize_primitives_json("./inputs/oriented_ellipsoid.json", "./outputs/m2_ellipsoid.mit")
    print(f"Generato m2_ellipsoid.mit")
    rasterize_primitives_json("./inputs/two_spheres.json", "./outputs/m3_two_spheres.mit")
    print(f"Generato m3_two_spheres.mit")

    print("Dimmi la path del comando mi_convert (mi_convert): ", end="")
    command = input()
    if command == "": command="mi_convert"
    print("Generating M1 OBJ...")
    os.system(f"{command} ./outputs/m1_sphere.mit ./outputs/m1_sphere.obj")
    print("Generating M2 OBJ...")
    os.system(f"{command} ./outputs/m2_ellipsoid.mit ./outputs/m2_ellipsoid.obj")
    print("Generating M3 OBJ...")
    os.system(f"{command} ./outputs/m3_two_spheres.mit ./outputs/m3_two_spheres.obj")
