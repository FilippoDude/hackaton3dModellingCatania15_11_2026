# =====================================
# 1. DEFINIZIONE DELLA MESH
# =====================================
import json
import os
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
    
    @classmethod
    def from_obj(cls, filepath: str) -> "Mesh":
        """Carica un file .obj e restituisce direttamente un'istanza di Mesh."""
        vertices = []
        faces = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("v "):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    parts = line.split()[1:]
                    # Gestisce sia 'f v1 v2 v3' che 'f v1/vt1/vn1 ...' e indici negativi
                    idx = []
                    for p in parts:
                        v_idx = int(p.split("/")[0])
                        idx.append(v_idx - 1 if v_idx > 0 else len(vertices) + v_idx)
                    
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
        return cls(vertices=vertices, faces=faces)

    @classmethod
    def from_json(cls, filepath: str) -> "list[Mesh]":
        with open(filepath, "r", encoding="utf-8") as f:
            objects = json.load(f)

        meshes = []
        for obj in objects:
            matrix = np.array(obj["matrix_world"], dtype=float)
            if matrix.shape != (4, 4):
                raise ValueError(f"{obj['name']}: matrix_world must be a 4x4 array")

            mesh = cls.from_obj(os.path.join(os.path.dirname(filepath), obj["name"] + ".obj"))
            mesh.vertices = mesh.vertices @ matrix[:3, :3].T + matrix[:3, 3]
            meshes.append(mesh)
        return meshes
