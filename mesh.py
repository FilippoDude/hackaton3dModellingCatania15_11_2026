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

    def save_obj(self, filepath):
        """Esporta la Mesh in formato .obj per la validazione."""
        with open(filepath, 'w') as f:
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in self.faces:
                # Gli indici .obj partono da 1
                f.write("f " + " ".join(str(i + 1) for i in face) + "\n")
