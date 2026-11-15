# =====================================
# 1. DEFINIZIONE DELLA MESH
# =====================================
import numpy as np
class Mesh:

    def __init__(self, vertices, faces):

        self.vertices = np.array(vertices, dtype=float)
        self.faces = np.array(faces, dtype=int)

    def get_face_vertices(self, face_index):

        # Recupera gli indici dei vertici della faccia
        indices = self.faces[face_index]

        # Recupera le coordinate XYZ
        return self.vertices[indices]