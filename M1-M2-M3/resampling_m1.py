import numpy as np


class MIStructure:
    """
    Struttura dati Marching Intersections (MI).
    Memorizza step, bounding box intero e gli intervalli [t_in, t_out]
    in coordinate di griglia per i 3 piani ortogonali (0=X, 1=Y, 2=Z).
    """
    def __init__(self, step: float, box: tuple[int, int, int, int, int, int]):
        self.step = float(step)
        self.box = tuple(int(b) for b in box)  # (min_x, min_y, min_z, max_x, max_y, max_z)
        # planes[axis][(row, col)] = [t_in_1, t_out_1, t_in_2, t_out_2, ...]
        self.planes: dict[int, dict[tuple[int, int], list[float]]] = {
            0: {},
            1: {},
            2: {}
        }

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
    """
    Intersezione analitica raggio-sfera: ||(O + t*D) - C||^2 = R^2.
    Restituisce i valori scalari t (in unità mondo) ordinati, oppure lista vuota.
    """
    oc = origin - center
    a = np.dot(direction, direction)
    b = 2.0 * np.dot(oc, direction)
    c = np.dot(oc, oc) - radius * radius

    discriminant = b * b - 4.0 * a * c
    if discriminant <= 0.0:
        return []

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    return [t1, t2]


def rasterize_sphere_m1(
    center=(0.0, 0.0, 0.0),
    radius=1.0,
    step_percentage=0.10,
    output_path="m1_sphere.mit"
) -> MIStructure:
    center = np.array(center, dtype=float)

    # Bounding box teorico della sfera [-R, R]^3
    bbox_min = center - radius
    bbox_max = center + radius

    # Step = 10% della diagonale del bounding box (2 * sqrt(3) * 0.1 = 0.34641016)
    diagonal = np.linalg.norm(bbox_max - bbox_min)
    step = diagonal * step_percentage

    # Indici interi della griglia centrata sull'origine (es. -4 .. +4)
    grid_min = np.floor(bbox_min / step).astype(int) - 1
    grid_max = np.ceil(bbox_max / step).astype(int) + 1
    box = (
        int(grid_min[0]), int(grid_min[1]), int(grid_min[2]),
        int(grid_max[0]), int(grid_max[1]), int(grid_max[2])
    )

    mi = MIStructure(step=step, box=box)

    # Numero di campionamenti per ciascun asse
    nx = box[3] - box[0] + 1
    ny = box[4] - box[1] + 1
    nz = box[5] - box[2] + 1

    # Campionamento sui 3 piani ortogonali
    for axis in (0, 1, 2):
        if axis == 0:
            # PLANE 0: raggi paralleli a X, indicizzati da (row=Y, col=Z)
            direction = np.array([1.0, 0.0, 0.0])
            for row in range(ny):
                y_world = (row + box[1]) * step
                for col in range(nz):
                    z_world = (col + box[2]) * step
                    origin = np.array([0.0, y_world, z_world])
                    hits = intersect_ray_sphere(origin, direction, center, radius)
                    if hits:
                        # Conversione in coordinate di griglia (diviso step)
                        mi.add_line(0, row, col, [t / step for t in hits])

        elif axis == 1:
            # PLANE 1: raggi paralleli a Y, indicizzati da (row=X, col=Z)
            direction = np.array([0.0, 1.0, 0.0])
            for row in range(nx):
                x_world = (row + box[0]) * step
                for col in range(nz):
                    z_world = (col + box[2]) * step
                    origin = np.array([x_world, 0.0, z_world])
                    hits = intersect_ray_sphere(origin, direction, center, radius)
                    if hits:
                        mi.add_line(1, row, col, [t / step for t in hits])

        elif axis == 2:
            # PLANE 2: raggi paralleli a Z, indicizzati da (row=X, col=Y)
            direction = np.array([0.0, 0.0, 1.0])
            for row in range(nx):
                x_world = (row + box[0]) * step
                for col in range(ny):
                    y_world = (col + box[1]) * step
                    origin = np.array([x_world, y_world, 0.0])
                    hits = intersect_ray_sphere(origin, direction, center, radius)
                    if hits:
                        mi.add_line(2, row, col, [t / step for t in hits])

    mi.save_mit(output_path)
    return mi

