import numpy as np


class MIStructure:
    def __init__(self, step: float, box: tuple[int, int, int, int, int, int]):
        self.step = float(step)
        self.box = tuple(int(b) for b in box)
        self.planes: dict[int, dict[tuple[int, int], list[float]]] = {0: {}, 1: {}, 2: {}}

    def add_line(self, plane: int, row: int, col: int, intersections: list[float]):
        if len(intersections) > 0:
            self.planes[plane][(row, col)] = sorted(float(t) for t in intersections)

    def save_mit(self, filepath: str):
        min_x, min_y, min_z, max_x, max_y, max_z = self.box
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# MI_FORMAT_VERSION 1.0\n")
            f.write(f"STEP {self.step:.9g}\n")
            f.write("# Integer grid box: min_x min_y min_z max_x max_y max_z\n")
            f.write(f"BOX {min_x} {min_y} {min_z} {max_x} {max_y} {max_z}\n\n")
            f.write("# Plane Data (e.g., Axis 0 = X Plane, Axis 1 = Y, Axis 2 = Z)\n")
            f.write("# Format per line: row column [intersection positions in grid coordinates...]\n")

            for axis in (0, 1, 2):
                f.write(f"PLANE {axis}\n")
                for (row, col), vals in sorted(self.planes[axis].items()):
                    vals_str = " ".join(f"{v:.9g}" for v in vals)
                    f.write(f"LINE {row} {col} {vals_str}\n")
                f.write("END_PLANE\n\n")


def intersect_ray_sphere(origin: np.ndarray, direction: np.ndarray, center: np.ndarray, radius: float) -> list[float]:
    oc = origin - center
    a = np.dot(direction, direction)
    b = 2.0 * np.dot(oc, direction)
    c = np.dot(oc, oc) - radius * radius

    discriminant = b * b - 4.0 * a * c
    if discriminant <= 0.0:
        return []

    sqrt_disc = np.sqrt(discriminant)
    return [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]


def rasterize_sphere_m1(
    center=(0.0, 0.0, 0.0),
    radius=1.0,
    step_percentage=0.10,
    output_path="m1_sphere.mit"
) -> MIStructure:
    center = np.array(center, dtype=float)
    bbox_min = center - radius
    bbox_max = center + radius

    step = np.linalg.norm(bbox_max - bbox_min) * step_percentage

    grid_min = np.floor(bbox_min / step).astype(int) - 1
    grid_max = np.ceil(bbox_max / step).astype(int) + 1
    box = (
        int(grid_min[0]), int(grid_min[1]), int(grid_min[2]),
        int(grid_max[0]), int(grid_max[1]), int(grid_max[2])
    )

    mi = MIStructure(step=step, box=box)
    min_x, min_y, min_z, max_x, max_y, max_z = box
    nx = max_x - min_x + 1
    ny = max_y - min_y + 1
    nz = max_z - min_z + 1

    # PLANE 0 (X): row = Y (0..ny-1), col = Z (0..nz-1)
    for row in range(ny):
        for col in range(nz):
            y_w = (row + min_y) * step
            z_w = (col + min_z) * step
            hits = intersect_ray_sphere(np.array([0.0, y_w, z_w]), np.array([1.0, 0.0, 0.0]), center, radius)
            if hits:
                mi.add_line(0, row, col, [t / step for t in hits])

    # PLANE 1 (Y): row = Z (0..nz-1), col = X (0..nx-1)  [convenzione ciclica Y->Z->X]
    for row in range(nz):
        for col in range(nx):
            z_w = (row + min_z) * step
            x_w = (col + min_x) * step
            hits = intersect_ray_sphere(np.array([x_w, 0.0, z_w]), np.array([0.0, 1.0, 0.0]), center, radius)
            if hits:
                mi.add_line(1, row, col, [t / step for t in hits])

    # PLANE 2 (Z): row = X (0..nx-1), col = Y (0..ny-1)  [convenzione ciclica Z->X->Y]
    for row in range(nx):
        for col in range(ny):
            x_w = (row + min_x) * step
            y_w = (col + min_y) * step
            hits = intersect_ray_sphere(np.array([x_w, y_w, 0.0]), np.array([0.0, 0.0, 1.0]), center, radius)
            if hits:
                mi.add_line(2, row, col, [t / step for t in hits])

    mi.save_mit(output_path)
    return mi
