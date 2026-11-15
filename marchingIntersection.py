



import numpy as np
import matplotlib.pyplot as plt
from mesh import Mesh
from converttomesh import mit_to_dexel_mesh

class SimulationConfig:

    # ==============================
    # VOLUME DI SIMULAZIONE
    # ==============================

    CUBE_SIZE = 2.0

    X_MIN = -1.0
    X_MAX =  1.0

    Y_MIN = -1.0
    Y_MAX =  1.0

    Z_MIN = -1.0
    Z_MAX =  1.0

    # ==============================
    # RAY SAMPLING
    # ==============================

    SAMPLING_RATE = 1

    # Numero di campioni per unità
    # di lunghezza su ciascun asse

    # ==============================
    # RAY PARAMETERS
    # ==============================

    RAY_LENGTH = 2.0

    # Direzione iniziale
    RAY_DIRECTION = np.array([0, 0, -1])

    # ==============================
    # VISUALIZATION
    # ==============================

    SHOW_RAYS = True
    SHOW_HITS = True
    SHOW_MESH = True



# =================================
# CREAZIONE DELLA MESH
# =================================

meshPlane = Mesh(

    vertices=[
        [-1, -1, 0.0],  # Vertex 0
        [ 1, -1, 0.0],  # Vertex 1
        [ 1,  1, 0.0],  # Vertex 2
        [-1,  1, 0.0]   # Vertex 3
    ],

    faces=[
        [0, 1, 2],  # Face 0
        [0, 2, 3]   # Face 1
    ]
)


meshCubo = Mesh(

    vertices=[
        # Faccia inferiore Z-
        [-1, -1, -1],  # 0
        [ 1, -1, -1],  # 1
        [ 1,  1, -1],  # 2
        [-1,  1, -1],  # 3

        # Faccia superiore Z+
        [-1, -1,  1],  # 4
        [ 1, -1,  1],  # 5
        [ 1,  1,  1],  # 6
        [-1,  1,  1]   # 7
    ],

    faces=[
        # Faccia inferiore (-Z)
        [0, 2, 1],
        [0, 3, 2],

        # Faccia superiore (+Z)
        [4, 5, 6],
        [4, 6, 7],

        # Faccia frontale (-Y)
        [0, 1, 5],
        [0, 5, 4],

        # Faccia posteriore (+Y)
        [3, 7, 6],
        [3, 6, 2],

        # Faccia sinistra (-X)
        [0, 4, 7],
        [0, 7, 3],

        # Faccia destra (+X)
        [1, 2, 6],
        [1, 6, 5]
    ]
)

meshTriangolo = Mesh(
    vertices=[
        [-1, -1, -1],  # 0 - base
        [ 1, -1, -1],  # 1 - base
        [ 1,  1, -1],  # 2 - base
        [-1,  1, -1],  # 3 - base
        [ 0,  0,  1]   # 4 - apice
    ],
    faces=[
        [0, 2, 1],
        [0, 3, 2],  # base

        [0, 1, 4],  # lato frontale
        [1, 2, 4],  # lato destro
        [2, 3, 4],  # lato posteriore
        [3, 0, 4]   # lato sinistro
    ]
)

meshEmptyInsideCube = Mesh(
    vertices = [
    # Cubo esterno: 0-7
    [-1, -1, -1],  # 0
    [ 1, -1, -1],  # 1
    [ 1,  1, -1],  # 2
    [-1,  1, -1],  # 3
    [-1, -1,  1],  # 4
    [ 1, -1,  1],  # 5
    [ 1,  1,  1],  # 6
    [-1,  1,  1],  # 7

    # Cubo interno: 8-15
    [-0.5, -0.5, -0.5],  # 8
    [ 0.5, -0.5, -0.5],  # 9
    [ 0.5,  0.5, -0.5],  # 10
    [-0.5,  0.5, -0.5],  # 11
    [-0.5, -0.5,  0.5],  # 12
    [ 0.5, -0.5,  0.5],  # 13
    [ 0.5,  0.5,  0.5],  # 14
    [-0.5,  0.5,  0.5]   # 15
],
faces = [
    # Cubo esterno
    [0, 2, 1], [0, 3, 2],       # -Z
    [4, 5, 6], [4, 6, 7],       # +Z
    [0, 1, 5], [0, 5, 4],       # -Y
    [3, 7, 6], [3, 6, 2],       # +Y
    [0, 4, 7], [0, 7, 3],       # -X
    [1, 2, 6], [1, 6, 5],       # +X

    # Cubo interno: orientamento invertito
    [8, 9, 10], [8, 10, 11],
    [12, 14, 13], [12, 15, 14],
    [8, 13, 9], [8, 12, 13],
    [11, 10, 14], [11, 14, 15],
    [8, 11, 15], [8, 15, 12],
    [9, 13, 14], [9, 14, 10]
]
)

def generate_rays(config):

    origins = []
    directions = []

    # Numero di campioni per faccia
    n = int(config.CUBE_SIZE * config.SAMPLING_RATE) + 1
    print('RIGA DI RAY PER FACCIA: ' + str(n))

    # Coordinate della griglia
    x = np.linspace(config.X_MIN, config.X_MAX, n)
    y = np.linspace(config.Y_MIN, config.Y_MAX, n)
    z = np.linspace(config.Z_MIN, config.Z_MAX, n)

    # =================================
    # FACE +X
    # Raggi diretti verso -X
    # =================================

    Y, Z = np.meshgrid(y, z, indexing="ij")

    points = np.column_stack((
        np.full(Y.size, config.X_MAX),
        Y.ravel(),
        Z.ravel()
    ))

    dirs = np.tile([-1, 0, 0], (len(points), 1))

    origins.extend(points)
    directions.extend(dirs)

    # =================================
    # FACE +Y
    # Raggi diretti verso -Y
    # =================================

    X, Z = np.meshgrid(x, z, indexing="ij")

    points = np.column_stack((
        X.ravel(),
        np.full(X.size, config.Y_MAX),
        Z.ravel()
    ))

    dirs = np.tile([0, -1, 0], (len(points), 1))

    origins.extend(points)
    directions.extend(dirs)

    # =================================
    # FACE +Z
    # Raggi diretti verso -Z
    # =================================

    X, Y = np.meshgrid(x, y, indexing="ij")

    points = np.column_stack((
        X.ravel(),
        Y.ravel(),
        np.full(X.size, config.Z_MAX)
    ))

    dirs = np.tile([0, 0, -1], (len(points), 1))

    origins.extend(points)
    directions.extend(dirs)

    return np.array(origins), np.array(directions)


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



def find_intersections(origins, directions, mesh):
    intersections = []

    for ray_id, (origin, direction) in enumerate(zip(origins, directions)):

        # Analizziamo tutte le facce della mesh
        for face_id in range(len(mesh.faces)):

            triangle = mesh.get_face_vertices(face_id)

            result = ray_triangle_intersection(
                origin, direction, triangle
            )

            if result is not None:
                hit_point, distance = result

                intersections.append({
                    "ray_id": ray_id,
                    "face_id": face_id,
                    "point": hit_point,
                    "distance": distance
                })

    # Ordiniamo le intersezioni per raggio e distanza
    intersections.sort(
        key=lambda hit: (hit["ray_id"], hit["distance"])
    )

    return intersections



# Generazione dei raggi
origins, directions = generate_rays(SimulationConfig)

imported_mesh = mit_to_dexel_mesh('./sphere.mit', 2)

# Calcolo delle intersezioni
hits = find_intersections(
    origins,
    directions,
    imported_mesh
)

# Risultati
print("Raggi generati:", len(origins))
print("Raggi che colpiscono la mesh:", len(hits))

for hit in hits[:10]:

    print(
        f"Ray {hit['ray_id']} | "
        f"Face {hit['face_id']} | "
        f"Point {hit['point']} | "
        f"Distance {hit['distance']:.3f}"
    )



def build_output_mesh(intersections, decimals=8):

    # Estraiamo i punti di intersezione
    points = [
        intersection["point"]
        for intersection in intersections
    ]

    # Se non ci sono intersezioni
    if len(points) == 0:
        return Mesh([], [])

    # Convertiamo in array NumPy
    points = np.array(points, dtype=float)

    # Eliminiamo i punti duplicati
    unique_vertices = np.unique(
        np.round(points, decimals=decimals),
        axis=0
    )

    # Creiamo una mesh senza facce
    output_mesh = Mesh(
        vertices=unique_vertices,
        faces=[]
    )

    return output_mesh


build_output_mesh(hits)
