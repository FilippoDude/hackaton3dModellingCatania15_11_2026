import numpy as np
from mesh import Mesh


def parse_mit_file(filepath):
    """Legge un file .mit e restituisce metadati e dizionario dei piani."""
    step = 0.0
    box = None
    planes = {0: [], 1: [], 2: []}
    current_plane = None

    with open(filepath, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            tag = parts[0]

            if tag == 'STEP':
                step = float(parts[1])
            elif tag == 'BOX':
                box = [int(x) for x in parts[1:7]]
            elif tag == 'PLANE':
                current_plane = int(parts[1])
            elif tag == 'END_PLANE':
                current_plane = None
            elif tag == 'LINE' and current_plane is not None:
                row = int(parts[1])
                col = int(parts[2])
                vals = [float(x) for x in parts[3:]]
                # Raggruppa a coppie [t_in, t_out]
                intervals = [(vals[i], vals[i+1]) for i in range(0, len(vals) - 1, 2)]
                planes[current_plane].append((row, col, intervals))

    return step, box, planes


def mit_to_dexel_mesh(filepath, use_plane=2):
    """
    Converte un file .mit nella struttura Mesh generando un parallelepipedo
    (8 vertici, 12 facce triangolari) per ogni intervallo [t_in, t_out].
    Non richiede librerie esterne oltre a NumPy.
    """
    step, box, planes = parse_mit_file(filepath)
    min_x, min_y, min_z, _, _, _ = box
    half = 0.5 * step

    all_vertices = []
    all_faces = []

    # Template topologia cubo (identico al tuo meshCubo)
    cube_faces_template = np.array([
        [0, 2, 1], [0, 3, 2],  # -Z
        [4, 5, 6], [4, 6, 7],  # +Z
        [0, 1, 5], [0, 5, 4],  # -Y
        [3, 7, 6], [3, 6, 2],  # +Y
        [0, 4, 7], [0, 7, 3],  # -X
        [1, 2, 6], [1, 6, 5]   # +X
    ], dtype=int)

    for row, col, intervals in planes[use_plane]:
        for t_in, t_out in intervals:
            # Conversione da coordinate griglia a coordinate mondo
            if use_plane == 0:      # Raggio lungo X: row=Y, col=Z
                y0, y1 = (row + min_y) * step - half, (row + min_y) * step + half
                z0, z1 = (col + min_z) * step - half, (col + min_z) * step + half
                x0, x1 = t_in * step, t_out * step
            elif use_plane == 1:    # Raggio lungo Y: row=X, col=Z
                x0, x1 = (row + min_x) * step - half, (row + min_x) * step + half
                z0, z1 = (col + min_z) * step - half, (col + min_z) * step + half
                y0, y1 = t_in * step, t_out * step
            else:                   # Raggio lungo Z: row=X, col=Y
                x0, x1 = (row + min_x) * step - half, (row + min_x) * step + half
                y0, y1 = (col + min_y) * step - half, (col + min_y) * step + half
                z0, z1 = t_in * step, t_out * step

            v_offset = len(all_vertices)
            box_verts = [
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
            ]
            all_vertices.extend(box_verts)
            all_faces.extend(cube_faces_template + v_offset)

    return Mesh(vertices=all_vertices, faces=all_faces)


def mit_to_smooth_mesh(filepath, supersample=2):
    """
    Ricostruisce una superficie chiusa e continua (triangolata via Marching Cubes)
    combinando tutti e 3 i piani ortogonali del file .mit per massima precisione.
    Richiede scikit-image (`pip install scikit-image`).
    """
    from skimage.measure import marching_cubes

    step, box, planes = parse_mit_file(filepath)
    min_x, min_y, min_z, max_x, max_y, max_z = box

    # Griglia volumetrica 3D ad alta risoluzione
    sub_step = step / supersample
    nx = (max_x - min_x) * supersample + 3
    ny = (max_y - min_y) * supersample + 3
    nz = (max_z - min_z) * supersample + 3

    origin = np.array([min_x - 1/supersample, min_y - 1/supersample, min_z - 1/supersample]) * step
    xs = origin[0] + np.arange(nx) * sub_step
    ys = origin[1] + np.arange(ny) * sub_step
    zs = origin[2] + np.arange(nz) * sub_step

    # Accumulatore di voto dai 3 piani (0, 1, 2)
    volume = np.zeros((nx, ny, nz), dtype=np.float32)

    for axis, lines in planes.items():
        for row, col, intervals in lines:
            for t_in, t_out in intervals:
                w_in, w_out = t_in * step, t_out * step
                if axis == 0:
                    iy = int(round(((row + min_y) * step - origin[1]) / sub_step))
                    iz = int(round(((col + min_z) * step - origin[2]) / sub_step))
                    mask = (xs >= w_in) & (xs <= w_out)
                    if 0 <= iy < ny and 0 <= iz < nz:
                        volume[mask, iy, iz] += 1.0
                elif axis == 1:
                    ix = int(round(((row + min_x) * step - origin[0]) / sub_step))
                    iz = int(round(((col + min_z) * step - origin[2]) / sub_step))
                    mask = (ys >= w_in) & (ys <= w_out)
                    if 0 <= ix < nx and 0 <= iz < nz:
                        volume[ix, mask, iz] += 1.0
                elif axis == 2:
                    ix = int(round(((row + min_x) * step - origin[0]) / sub_step))
                    iy = int(round(((col + min_y) * step - origin[1]) / sub_step))
                    mask = (zs >= w_in) & (zs <= w_out)
                    if 0 <= ix < nx and 0 <= iy < ny:
                        volume[ix, iy, mask] += 1.0

    # Soglia 1.5 su 3 piani (voto di maggioranza)
    verts, faces, _, _ = marching_cubes(volume, level=1.5, spacing=(sub_step, sub_step, sub_step))
    verts += origin
    return Mesh(vertices=verts, faces=faces)


def obj_to_dexel_mesh(filepath):
    """
    Converte un file .obj nella struttura Mesh.
    Supporta vertici (v) e facce (f) triangolari o poligonali.
    Non richiede librerie esterne oltre a NumPy.
    """
    all_vertices = []
    all_faces = []

    with open(filepath, "r") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if parts[0] == "v":
                # Coordinate del vertice
                vertex = [float(value) for value in parts[1:4]]
                all_vertices.append(vertex)

            elif parts[0] == "f":
                # Indici dei vertici della faccia
                face = []

                for value in parts[1:]:
                    # Gestisce anche formati come 1/2/3 o 1//3
                    vertex_index = int(value.split("/")[0])

                    # Gli indici OBJ partono da 1.
                    # Gli indici negativi sono relativi alla fine della lista.
                    if vertex_index > 0:
                        vertex_index -= 1
                    else:
                        vertex_index = len(all_vertices) + vertex_index

                    face.append(vertex_index)

                all_faces.append(face)

    return Mesh(
        vertices=np.array(all_vertices, dtype=float),
        faces=np.array(all_faces, dtype=object)
    )