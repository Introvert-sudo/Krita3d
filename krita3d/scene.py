"""3D scene management: mesh loading and per-node transform tracking."""
from __future__ import annotations

import numpy as np

try:
    import trimesh
    import trimesh.transformations as tf
    TRIMESH_OK = True
except ImportError:
    TRIMESH_OK = False

# Colour palette for assigning colours to meshes without material data
_PALETTE = [
    (0.70, 0.75, 0.82),
    (0.82, 0.68, 0.55),
    (0.55, 0.75, 0.65),
    (0.80, 0.72, 0.55),
    (0.65, 0.65, 0.80),
    (0.80, 0.60, 0.65),
]


def euler_to_matrix(rx, ry, rz, tx, ty, tz):
    """Row-major 4x4 transform from Euler degrees + translation."""
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    T = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]], dtype=np.float32)
    return T @ Rz @ Ry @ Rx


class Mesh3D:
    def __init__(self, name, vertices, normals, indices, color=None):
        self.name = name
        self.vertices = vertices.astype(np.float32)
        self.normals = normals.astype(np.float32)
        self.indices = indices.astype(np.uint32)
        self.base_transform = np.eye(4, dtype=np.float32)
        self.color = np.array(color or _PALETTE[0], dtype=np.float32)
        # User pose: (rx, ry, rz, tx, ty, tz) in degrees/units
        self.pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def world_transform(self):
        """pose_matrix @ base_transform — applied around mesh bounding-box centre."""
        centre = self.vertices.mean(axis=0)
        # Translate to centre, apply pose rotation, translate back
        T_to = np.eye(4, dtype=np.float32)
        T_to[0, 3] = -centre[0]; T_to[1, 3] = -centre[1]; T_to[2, 3] = -centre[2]
        T_from = np.eye(4, dtype=np.float32)
        T_from[0, 3] = centre[0]; T_from[1, 3] = centre[1]; T_from[2, 3] = centre[2]
        pose_mat = euler_to_matrix(*self.pose)
        local_pose = T_from @ pose_mat @ T_to
        return local_pose @ self.base_transform


class Scene3D:
    def __init__(self):
        self.meshes: list[Mesh3D] = []

    def clear(self):
        self.meshes.clear()

    def load(self, path: str) -> int:
        if not TRIMESH_OK:
            raise RuntimeError("trimesh is not installed.\nRun: sudo pacman -S python-trimesh")

        loaded = trimesh.load(path, force='scene')
        if isinstance(loaded, trimesh.Trimesh):
            scene = trimesh.Scene()
            scene.add_geometry(loaded, node_name="mesh_0", geom_name="mesh_0")
        elif isinstance(loaded, trimesh.Scene):
            scene = loaded
        else:
            raise ValueError(f"Unsupported file type: {type(loaded)}")

        self.meshes.clear()
        palette_idx = 0

        for node_name in scene.graph.nodes_geometry:
            try:
                T, geom_name = scene.graph[node_name]
            except Exception:
                continue
            if geom_name not in scene.geometry:
                continue
            geom = scene.geometry[geom_name]
            if not isinstance(geom, trimesh.Trimesh):
                continue
            if len(geom.faces) == 0:
                continue

            verts = np.array(geom.vertices, dtype=np.float32)
            norms = np.array(geom.vertex_normals, dtype=np.float32)
            idx = np.array(geom.faces, dtype=np.uint32).flatten()

            # Try to get material colour
            color = _PALETTE[palette_idx % len(_PALETTE)]
            try:
                c = geom.visual.main_color
                if c is not None and len(c) >= 3:
                    color = (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
            except Exception:
                pass

            mesh = Mesh3D(node_name, verts, norms, idx, color)
            mesh.base_transform = np.array(T, dtype=np.float32)
            self.meshes.append(mesh)
            palette_idx += 1

        return len(self.meshes)

    def node_names(self):
        return [m.name for m in self.meshes]

    def get_mesh(self, name) -> Mesh3D | None:
        for m in self.meshes:
            if m.name == name:
                return m
        return None

    def bounds(self):
        """(centre_xyz, radius) for placing the camera."""
        if not self.meshes:
            return np.zeros(3, dtype=np.float32), 1.0
        pts = []
        for m in self.meshes:
            T = m.world_transform()
            v = np.hstack([m.vertices, np.ones((len(m.vertices), 1), dtype=np.float32)])
            pts.append((T @ v.T).T[:, :3])
        all_pts = np.vstack(pts)
        centre = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2.0
        radius = float(np.linalg.norm(all_pts - centre, axis=1).max())
        return centre.astype(np.float32), max(radius, 0.001)
