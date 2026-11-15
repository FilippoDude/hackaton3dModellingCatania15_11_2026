"""Marching Intersections: struttura dati MI, file .mit e mesh duale di verifica.

Struttura MI = (step, box, planes):
  step   passo della griglia (le linee stanno sui multipli interi di step, a partire da 0)
  box    [min_x, min_y, min_z, max_x, max_y, max_z] in coordinate intere di griglia
  planes planes[axis][(row, col)] = intersezioni ordinate lungo il raggio, in coordinate
         di griglia, con row/col relativi al minimo del box. Convenzione ciclica di
         mi_convert: asse 0: row=Z col=Y; asse 1: row=X col=Z; asse 2: row=Y col=X.
"""
import numpy as np

from mesh import Mesh

# Distanza minima (in unita' di griglia) tra un'intersezione e un nodo della griglia
NODE_EPS = 1e-4


def plane_axes(axis):
    """Assi (u, v) del piano perpendicolare ad `axis`, in ordine ciclico.
    Nel file .mit: col = u - min_u, row = v - min_v."""
    return (axis + 1) % 3, (axis + 2) % 3


def away_from_nodes(values):
    """Allontana di NODE_EPS dai nodi della griglia le intersezioni che ci cadono (quasi)
    sopra, considerando il nodo sempre FUORI dal solido: un'entrata (indice pari) va dopo
    il nodo, un'uscita (indice dispari) prima. Cosi' il nodo risulta fuori lungo tutti e
    tre gli assi (altrimenti mi_convert segnala 'Occupancy disagrees').
    Gli intervalli che si annullano vengono tolti."""
    values = np.asarray(values, dtype=float)
    nodes = np.round(values)
    shift = np.where(np.arange(len(values)) % 2 == 0, NODE_EPS, -NODE_EPS)
    values = np.where(np.abs(values - nodes) < NODE_EPS, nodes + shift, values)
    pairs = [(t_in, t_out) for t_in, t_out in zip(values[::2], values[1::2]) if t_out > t_in]
    return one_per_edge([t for pair in pairs for t in pair])


def one_per_edge(values):
    """Al massimo un'intersezione per spigolo di griglia: due intersezioni consecutive
    tra gli stessi due nodi (un intervallo o un buco piu' sottile di una cella) non
    cambiano lo stato di nessun nodo, quindi si annullano a vicenda."""
    kept = []
    for t in sorted(values):
        if kept and np.floor(kept[-1]) == np.floor(t):
            kept.pop()
        else:
            kept.append(float(t))
    return kept


def occupancy(box, planes):
    """Per ogni asse, matrice booleana dei nodi della griglia che risultano dentro il
    solido secondo le linee di quell'asse. Indici relativi al minimo del box."""
    lo, hi = np.array(box[:3]), np.array(box[3:])
    occ = np.zeros((3, *(hi - lo)), dtype=bool)
    for axis in range(3):
        a, b = plane_axes(axis)
        for (row, col), values in planes[axis].items():
            index = [0, 0, 0]
            index[a], index[b] = col, row
            for t_in, t_out in zip(values[::2], values[1::2]):
                k0 = max(int(np.ceil(t_in)) - lo[axis], 0)
                k1 = min(int(np.floor(t_out)) - lo[axis], hi[axis] - lo[axis] - 1)
                if k0 <= k1:
                    index[axis] = slice(k0, k1 + 1)
                    occ[axis][tuple(index)] = True
    return occ


def enforce_consistency(box, planes):
    """Rende coerente il volume: ogni nodo deve risultare dentro (o fuori) lungo tutti e
    tre gli assi, altrimenti mi_convert lo rifiuta ('Occupancy disagrees'). Succede ai
    nodi quasi sulla superficie, dove gli errori numerici dei tre assi non concordano.
    Si decide a maggioranza e sull'asse in disaccordo si sposta oltre il nodo, di
    NODE_EPS, l'intersezione piu' vicina (o si aggiunge un intervallo minimo attorno al
    nodo se non ce n'e' una entro una cella). Ritorna il numero di correzioni."""
    lo = np.array(box[:3])
    occ = occupancy(box, planes)
    majority = occ.sum(axis=0) >= 2
    fixes = 0
    for node in np.argwhere((occ != majority).any(axis=0)):
        for axis in np.flatnonzero(occ[(slice(None), *node)] != majority[tuple(node)]):
            a, b = plane_axes(axis)
            key = (int(node[b]), int(node[a]))
            k = node[axis] + lo[axis]
            values = list(planes[axis].get(key, []))
            nearest = min(range(len(values)), key=lambda n: abs(values[n] - k), default=None)
            if nearest is not None and abs(values[nearest] - k) < 1:
                # Portando l'intersezione dall'altra parte del nodo cambia solo lui
                values[nearest] = k + NODE_EPS if values[nearest] < k else k - NODE_EPS
            else:
                values += [k - NODE_EPS, k + NODE_EPS]
            planes[axis][key] = one_per_edge(values)
            if not planes[axis][key]:
                del planes[axis][key]
            fixes += 1
    return fixes


def ray_triangle_intersection(origin, direction, triangle):

    v0, v1, v2 = triangle

    epsilon = 1e-8

    edge1 = v1 - v0
    edge2 = v2 - v0

    h = np.cross(direction, edge2)
    a = np.dot(edge1, h)

    # Raggio parallelo al triangolo
    if abs(a) < epsilon:
        return None

    f = 1.0 / a
    s = origin - v0

    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return None

    q = np.cross(s, edge1)
    v = f * np.dot(direction, q)

    if v < 0.0 or u + v > 1.0:
        return None

    t = f * np.dot(edge2, q)

    # Consideriamo solo intersezioni davanti al raggio
    if t > epsilon:

        hit_point = origin + t * direction

        return hit_point, t

    return None


def mesh_object(mesh):
    """Mesh di triangoli chiusa, intersecata con ray_triangle_intersection."""
    triangles = [mesh.get_face_vertices(f) for f in range(len(mesh.faces))]

    def hits(origin, direction):
        results = (ray_triangle_intersection(origin, direction, t) for t in triangles)
        # shortcut: regola pari/dispari con dedup per arrotondamento; un raggio tangente
        # alla mesh puo' dare un conteggio dispari (l'ultima intersezione viene scartata),
        # passare alle intersezioni orientate (segno di a) se succede
        return sorted({round(r[1], 8) for r in results if r is not None})

    return hits, (mesh.vertices.min(axis=0), mesh.vertices.max(axis=0))


def rasterize(objects, percent=10):
    """Crea la struttura MI dell'unione degli oggetti (sphere_object / mesh_object).
    Il passo e' percent% della diagonale del bounding box complessivo."""
    lo = np.min([bounds[0] for _, bounds in objects], axis=0)
    hi = np.max([bounds[1] for _, bounds in objects], axis=0)
    step = np.linalg.norm(hi - lo) * percent / 100
    lo = np.floor(lo / step).astype(int) - 1
    hi = np.ceil(hi / step).astype(int) + 1

    planes = {}
    for axis in range(3):
        a, b = plane_axes(axis)
        direction = np.eye(3)[axis]
        lines = planes[axis] = {}

        for i in range(lo[a], hi[a] + 1):
            for j in range(lo[b], hi[b] + 1):
                origin = np.zeros(3)
                origin[[axis, a, b]] = lo[axis] * step, i * step, j * step

                intervals = []
                for hits, _ in objects:
                    distances = hits(origin, direction)
                    intervals += zip(distances[::2], distances[1::2])

                # Unione degli intervalli [entrata, uscita] che si sovrappongono
                merged = []
                for t_in, t_out in sorted(intervals):
                    if merged and t_in <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], t_out)
                    else:
                        merged.append([t_in, t_out])

                values = away_from_nodes([lo[axis] + t / step for pair in merged for t in pair])
                if values:
                    lines[(j - lo[b], i - lo[a])] = values

    box = [int(x) for x in (*lo, *hi)]
    enforce_consistency(box, planes)
    return step, box, planes


def rasterize_mesh(mesh, percent=10):
    """Crea la struttura MI di una mesh chiusa di triangoli rasterizzando ogni triangolo
    sulle linee di griglia che attraversano la sua proiezione.
    Il passo e' percent% della diagonale del bounding box."""
    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    step = np.linalg.norm(hi - lo) * percent / 100
    lo = np.floor(lo / step).astype(int) - 1
    hi = np.ceil(hi / step).astype(int) + 1
    triangles = (mesh.vertices[np.array(mesh.faces)] / step).tolist()  # coordinate di griglia
    eps = 1e-9

    planes = {}
    for axis in range(3):
        a, b = plane_axes(axis)

        hits = []  # righe (i, j, profondita', verso del triangolo rispetto al raggio)
        for p0, p1, p2 in triangles:
            e1a, e1b, e2a, e2b = p1[a] - p0[a], p1[b] - p0[b], p2[a] - p0[a], p2[b] - p0[b]
            area = e1a * e2b - e1b * e2a
            if abs(area) < 1e-12:  # triangolo parallelo al raggio
                continue

            # Linee di griglia dentro il rettangolo che contiene la proiezione del triangolo
            i = np.arange(np.ceil(min(p0[a], p1[a], p2[a]) - eps), np.floor(max(p0[a], p1[a], p2[a]) + eps) + 1)
            j = np.arange(np.ceil(min(p0[b], p1[b], p2[b]) - eps), np.floor(max(p0[b], p1[b], p2[b]) + eps) + 1)
            if len(i) == 0 or len(j) == 0:
                continue
            i, j = np.meshgrid(i, j, indexing="ij")

            # Coordinate baricentriche della linea nella proiezione del triangolo
            u = ((i - p0[a]) * e2b - (j - p0[b]) * e2a) / area
            v = ((j - p0[b]) * e1a - (i - p0[a]) * e1b) / area
            inside = (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps)
            if not inside.any():
                continue
            depth = p0[axis] + u[inside] * (p1[axis] - p0[axis]) + v[inside] * (p2[axis] - p0[axis])
            hits.append(np.column_stack((i[inside], j[inside], depth.round(9), np.full(len(depth), np.sign(area)))))

        lines = planes[axis] = {}
        if not hits:
            continue

        # Ordina per linea e profondita' e toglie lo stesso punto visto da triangoli adiacenti
        hits = np.unique(np.concatenate(hits), axis=0)
        # Entrata e uscita nello stesso punto (raggio radente a uno spigolo): si annullano
        grazing = (hits[1:, :3] == hits[:-1, :3]).all(axis=1)
        keep = np.ones(len(hits), dtype=bool)
        keep[1:] &= ~grazing
        keep[:-1] &= ~grazing
        hits = hits[keep]

        line_ids, starts = np.unique(hits[:, :2], axis=0, return_index=True)
        odd = 0
        for (i, j), values in zip(line_ids.astype(int), np.split(hits[:, 2], starts[1:])):
            # shortcut: regola pari/dispari; su una linea con conteggio dispari (mesh non
            # chiusa o caso degenere) l'ultima intersezione viene scartata, usare il verso
            # dei triangoli per ricostruire gli intervalli se l'avviso qui sotto compare
            odd += len(values) % 2
            values = away_from_nodes(values[:len(values) // 2 * 2])
            if values:
                lines[(j - lo[b], i - lo[a])] = values
        if odd:
            print(f"Attenzione: asse {axis}, {odd} linee con un numero dispari di intersezioni")

    box = [int(x) for x in (*lo, *hi)]
    fixes = enforce_consistency(box, planes)
    if fixes:
        print(f"Coerenza: {fixes} intersezioni corrette vicino ai nodi della griglia")
    return step, box, planes


def save_mit(filepath, step, box, planes):
    with open(filepath, "w") as f:
        f.write("# MI_FORMAT_VERSION 1.0\n")
        f.write(f"STEP {step:.9g}\n")
        f.write("# Integer grid box: min_x min_y min_z max_x max_y max_z\n")
        f.write("BOX " + " ".join(str(x) for x in box) + "\n\n")
        f.write("# Plane Data (e.g., Axis 0 = X Plane, Axis 1 = Y, Axis 2 = Z)\n")
        f.write("# Format per line: row column [intersection positions in grid coordinates...]\n")
        for axis in range(3):
            f.write(f"PLANE {axis}\n")
            for (row, col), values in sorted(planes[axis].items()):  # ordine lessicografico
                f.write(f"LINE {row} {col} " + " ".join(f"{v:.9g}" for v in values) + "\n")
            f.write("END_PLANE\n\n")


def mi_to_mesh(step, box, planes):
    """Mesh duale: un vertice per ogni cella attraversata dalla superficie (media delle
    intersezioni sui suoi spigoli), un quad per ogni intersezione."""
    cell_points = {}  # cella (ix, iy, iz) -> intersezioni sui suoi spigoli
    quads = []

    for axis in range(3):
        a, b = plane_axes(axis)
        for (row, col), values in planes[axis].items():
            i, j = col + box[a], row + box[b]
            for n, value in enumerate(values):
                point = np.zeros(3)
                point[[axis, a, b]] = value * step, i * step, j * step

                # Le 4 celle che condividono lo spigolo di griglia intersecato
                quad = []
                for da, db in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
                    cell = [0, 0, 0]
                    cell[axis], cell[a], cell[b] = int(np.floor(value)), i + da, j + db
                    cell_points.setdefault(tuple(cell), []).append(point)
                    quad.append(tuple(cell))

                # n pari = il raggio entra nel solido: normale opposta al raggio
                quads.append(quad[::-1] if n % 2 == 0 else quad)

    index = {cell: n for n, cell in enumerate(cell_points)}
    vertices = [np.mean(points, axis=0) for points in cell_points.values()]
    return Mesh(vertices, [[index[cell] for cell in quad] for quad in quads])
