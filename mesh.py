# =====================================
# 1. DEFINIZIONE DELLA MESH
# =====================================
import numpy as np
class Mesh:

    def __init__(self, vertices, faces):
        self.vertices = np.array(vertices, dtype=float)
        self.faces = [np.array(face, dtype=int) for face in faces]

    def get_face_vertices(self, face_index):
        indices = self.faces[face_index]
        return self.vertices[indices]

    def save_obj(self, filepath):
        with open(filepath, 'w') as f:
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            for face in self.faces:
                f.write("f " + " ".join(str(i + 1) for i in face) + "\n")