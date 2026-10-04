"""Interactive 3D viewport using QOpenGLWidget + PyOpenGL."""
from __future__ import annotations

import ctypes
import numpy as np

try:
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    from PyQt6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram
    from PyQt6.QtGui import QSurfaceFormat, QOpenGLContext, QCursor
    from PyQt6.QtCore import Qt, pyqtSignal, QPoint
    _CORE_PROFILE = QSurfaceFormat.OpenGLContextProfile.CoreProfile
    _VERTEX   = QOpenGLShader.ShaderTypeBit.Vertex
    _FRAGMENT = QOpenGLShader.ShaderTypeBit.Fragment
    _CROSS_CURSOR = Qt.CursorShape.CrossCursor
    _ARROW_CURSOR = Qt.CursorShape.ArrowCursor
    _SIZE_CURSOR  = Qt.CursorShape.SizeAllCursor
    _SHIFT        = Qt.KeyboardModifier.ShiftModifier
except ImportError:
    from PyQt5.QtWidgets import QOpenGLWidget
    from PyQt5.QtGui import QOpenGLShader, QOpenGLShaderProgram, QSurfaceFormat, QOpenGLContext, QCursor
    from PyQt5.QtCore import Qt, pyqtSignal, QPoint
    _CORE_PROFILE = QSurfaceFormat.CoreProfile
    _VERTEX   = QOpenGLShader.Vertex
    _FRAGMENT = QOpenGLShader.Fragment
    _CROSS_CURSOR = Qt.CrossCursor
    _ARROW_CURSOR = Qt.ArrowCursor
    _SIZE_CURSOR  = Qt.SizeAllCursor
    _SHIFT        = Qt.ShiftModifier

try:
    from OpenGL.GL import (
        glClear, glClearColor, glEnable, glPolygonMode, glViewport,
        glBlendFunc, glGenVertexArrays, glBindVertexArray, glDeleteVertexArrays,
        glGenBuffers, glBindBuffer, glBufferData, glDeleteBuffers,
        glVertexAttribPointer, glEnableVertexAttribArray,
        glDrawElements, glGetUniformLocation,
        glUniformMatrix4fv, glUniform3fv, glUniform4fv, glUniform1f,
        GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
        GL_FLOAT, GL_FALSE, GL_TRUE, GL_ARRAY_BUFFER,
        GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW,
        GL_UNSIGNED_INT, GL_TRIANGLES, GL_FRONT_AND_BACK,
        GL_LINE, GL_FILL, GL_BLEND,
        GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    )
    OPENGL_OK = True
except ImportError:
    OPENGL_OK = False

from .scene import Scene3D


# ── Workaround: PyOpenGL picks EGL on Wayland, Krita uses XWayland/GLX ───
def _install_context_bridge():
    from OpenGL import platform as glplatform
    native = glplatform.GetCurrentContext

    def current_context():
        ctx = native()
        if ctx:
            return ctx
        qctx = QOpenGLContext.currentContext()
        return id(qctx) if qctx is not None else None

    glplatform.GetCurrentContext = current_context
    glplatform.PLATFORM.GetCurrentContext = current_context


if OPENGL_OK:
    _install_context_bridge()


# ── Tool modes ────────────────────────────────────────────────────────────
class ToolMode:
    ORBIT  = 0
    ROTATE = 1
    MOVE   = 2


# ── GLSL 330 core ─────────────────────────────────────────────────────────
_VERT = """\
#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_norm;
uniform mat4 u_mvp;
uniform mat4 u_model;
out vec3 v_norm;
out vec3 v_pos;
void main() {
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    mat3 nm = transpose(inverse(mat3(u_model)));
    v_norm  = normalize(nm * a_norm);
    v_pos   = vec3(u_model * vec4(a_pos, 1.0));
}
"""

_FRAG = """\
#version 330 core
in vec3 v_norm;
in vec3 v_pos;
uniform vec3  u_light;
uniform vec4  u_color;
uniform float u_selected;   // 1.0 = highlighted, 0.0 = normal
out vec4 frag_color;
void main() {
    vec3  n   = normalize(v_norm);
    float d   = max(dot(n, u_light), 0.0);
    float lit = 0.30 + 0.65 * d;
    float hl  = 1.0 + 0.28 * u_selected;
    frag_color = vec4(u_color.rgb * lit * hl, u_color.a);
}
"""


# ── Matrix math (row-major; passed to GL with GL_TRUE = transpose) ────────
def _perspective(fov_deg: float, aspect: float, near: float, far: float):
    f  = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    nf = near - far
    return np.array([
        [f / aspect, 0,  0,                    0],
        [0,          f,  0,                    0],
        [0,          0,  (far + near) / nf,    2 * far * near / nf],
        [0,          0, -1,                    0],
    ], dtype=np.float32)


def _lookat(eye, center, up):
    f = center - eye;  f /= np.linalg.norm(f)
    s = np.cross(f, up / np.linalg.norm(up));  s /= np.linalg.norm(s)
    u = np.cross(s, f)
    return np.array([
        [ s[0],  s[1],  s[2], -np.dot(s, eye)],
        [ u[0],  u[1],  u[2], -np.dot(u, eye)],
        [-f[0], -f[1], -f[2],  np.dot(f, eye)],
        [0,      0,     0,     1],
    ], dtype=np.float32)


# ── Möller-Trumbore batch (vectorized numpy) ──────────────────────────────
def _ray_mesh_hit(ro, rd_unnorm, vertices, triangles):
    """
    Vectorized ray-triangle intersection (Möller-Trumbore).

    ro          : (3,) ray origin in mesh-local space
    rd_unnorm   : (3,) ray direction in mesh-local space, NOT normalised
                  (t values correspond to world-space distances when the
                   ray was un-normalised using inv(world_T))
    vertices    : (V, 3) float32
    triangles   : (F, 3) int  — face vertex indices

    Returns     : minimum t (world-space distance) or np.inf if no hit
    """
    eps = 1e-7
    v0 = vertices[triangles[:, 0]]   # (F, 3)
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    e1 = v1 - v0                     # (F, 3)
    e2 = v2 - v0

    # h = cross(rd, e2) for every triangle
    h = np.cross(rd_unnorm, e2)      # broadcast (3,) × (F,3) → (F, 3)
    a = np.einsum('ij,ij->i', e1, h) # (F,) = dot(e1, h)

    valid = np.abs(a) > eps
    f = np.where(valid, 1.0 / a, 0.0)

    s  = ro - v0                     # (F, 3)
    u  = f * np.einsum('ij,ij->i', s, h)
    valid &= (u >= 0) & (u <= 1)

    q  = np.cross(s, e1)             # (F, 3)
    v  = f * (q @ rd_unnorm)         # (F,)  = f * dot(q_i, rd)
    valid &= (v >= 0) & (u + v <= 1)

    t  = f * np.einsum('ij,ij->i', e2, q)
    valid &= (t > eps)

    t_vals = np.where(valid, t, np.inf)
    return float(t_vals.min())


# ── GPU buffer wrapper ────────────────────────────────────────────────────
class _MeshGPU:
    __slots__ = ('vao', 'vbo', 'ibo', 'index_count')

    def __init__(self):
        self.vao = self.vbo = self.ibo = None
        self.index_count = 0

    def upload(self, vertices, normals, indices):
        self.free()
        data = np.hstack([vertices, normals]).flatten().astype(np.float32)
        idx  = indices.flatten().astype(np.uint32)

        self.vao = int(glGenVertexArrays(1))
        self.vbo = int(glGenBuffers(1))
        self.ibo = int(glGenBuffers(1))

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)

        stride = 6 * 4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)
        self.index_count = len(idx)

    def draw(self):
        if self.vao is None:
            return
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def free(self):
        if self.vao is not None:
            glDeleteVertexArrays(1, [self.vao])
            glDeleteBuffers(1, [self.vbo])
            glDeleteBuffers(1, [self.ibo])
            self.vao = self.vbo = self.ibo = None


# ── Viewport widget ───────────────────────────────────────────────────────
class Viewport3D(QOpenGLWidget):
    camera_changed    = pyqtSignal()
    selection_changed = pyqtSignal(str)   # emitted with mesh name when selection changes
    pose_changed      = pyqtSignal(str)   # emitted with mesh name after pose drag

    def __init__(self, scene: Scene3D, parent=None):
        super().__init__(parent)

        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(_CORE_PROFILE)
        fmt.setDepthBufferSize(24)
        fmt.setAlphaBufferSize(8)        # transparent background support
        fmt.setSamples(4)
        self.setFormat(fmt)

        self.scene = scene
        self._gpu: dict[str, _MeshGPU] = {}
        self._prog: QOpenGLShaderProgram | None = None

        # Camera
        self.azimuth   = 30.0
        self.elevation = 20.0
        self.distance  = 3.0
        self.target    = np.zeros(3, dtype=np.float32)
        self.fov       = 45.0
        self.wireframe = False

        # Tool state
        self.tool_mode: int = ToolMode.ORBIT
        self._selected_mesh: str | None = None

        # Drag state
        self._last_pos      = QPoint()
        self._dragging      = False      # left-drag in ORBIT mode (or fallback)
        self._pan_dragging  = False      # middle-drag
        self._pose_dragging = False      # left-drag in ROTATE/MOVE mode on a mesh

        self._needs_rebuild = False

        self.setMinimumSize(200, 150)

    # ── Public API ────────────────────────────────────────────────────────
    def set_scene(self, scene: Scene3D):
        self.scene = scene
        self._selected_mesh = None
        self._needs_rebuild = True
        self.reset_view()

    def rebuild(self):
        self.update()

    def reset_view(self):
        centre, radius = self.scene.bounds()
        self.target    = centre.copy()
        self.distance  = radius * 3.0
        self.azimuth   = 30.0
        self.elevation = 20.0
        self.update()

    def set_wireframe(self, on: bool):
        self.wireframe = on
        self.update()

    def set_tool_mode(self, mode: int):
        self.tool_mode = mode
        cursors = {
            ToolMode.ORBIT:  _ARROW_CURSOR,
            ToolMode.ROTATE: _CROSS_CURSOR,
            ToolMode.MOVE:   _SIZE_CURSOR,
        }
        self.setCursor(QCursor(cursors.get(mode, _ARROW_CURSOR)))

    def set_selected(self, name: str | None):
        self._selected_mesh = name
        self.update()

    # ── QOpenGLWidget overrides ───────────────────────────────────────────
    def initializeGL(self):
        if not OPENGL_OK:
            return
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.0, 0.0, 0.0, 0.0)   # fully transparent background

        self._prog = QOpenGLShaderProgram(self)
        ok = self._prog.addShaderFromSourceCode(_VERTEX,   _VERT)
        if not ok:
            print("[krita3d] Vertex shader error:", self._prog.log())
        ok = self._prog.addShaderFromSourceCode(_FRAGMENT, _FRAG)
        if not ok:
            print("[krita3d] Fragment shader error:", self._prog.log())
        if not self._prog.link():
            print("[krita3d] Shader link error:", self._prog.log())

        self._rebuild_gpu_buffers()
        self._needs_rebuild = False

    def resizeGL(self, w, h):
        glViewport(0, 0, w, max(h, 1))

    def paintGL(self):
        if not OPENGL_OK or self._prog is None:
            return
        if self._needs_rebuild:
            self._rebuild_gpu_buffers()
            self._needs_rebuild = False

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if self.wireframe else GL_FILL)

        w, h  = self.width(), max(self.height(), 1)
        proj  = _perspective(self.fov, w / h, 0.001, 10000.0)
        eye   = self._eye()
        view  = _lookat(eye, self.target, np.array([0, 1, 0], dtype=np.float32))

        self._prog.bind()
        pid = self._prog.programId()

        light = np.array([0.6, 0.8, 0.5], dtype=np.float32)
        light /= np.linalg.norm(light)
        glUniform3fv(glGetUniformLocation(pid, "u_light"), 1, light)

        loc_mvp  = glGetUniformLocation(pid, "u_mvp")
        loc_mdl  = glGetUniformLocation(pid, "u_model")
        loc_col  = glGetUniformLocation(pid, "u_color")
        loc_sel  = glGetUniformLocation(pid, "u_selected")

        for mesh in self.scene.meshes:
            gpu = self._gpu.get(mesh.name)
            if gpu is None:
                continue
            model = mesh.world_transform()
            mvp   = proj @ view @ model
            glUniformMatrix4fv(loc_mvp, 1, GL_TRUE, mvp.flatten())
            glUniformMatrix4fv(loc_mdl, 1, GL_TRUE, model.flatten())
            glUniform4fv(loc_col, 1, np.append(mesh.color, 1.0).astype(np.float32))
            is_sel = 1.0 if mesh.name == self._selected_mesh else 0.0
            glUniform1f(loc_sel, is_sel)
            gpu.draw()

        self._prog.release()

    # ── Mouse ─────────────────────────────────────────────────────────────
    def mousePressEvent(self, e):
        self._last_pos = e.pos()
        try:
            left = Qt.MouseButton.LeftButton
            mid  = Qt.MouseButton.MiddleButton
        except AttributeError:
            left = Qt.LeftButton
            mid  = Qt.MiddleButton

        if e.button() == mid:
            self._pan_dragging = True
            return

        if e.button() != left:
            return

        if self.tool_mode == ToolMode.ORBIT:
            self._dragging = True
        else:
            # Try to pick a mesh; fall back to orbit if nothing hit
            name = self._pick_mesh(e.pos().x(), e.pos().y())
            if name is not None:
                self._pose_dragging = True
                if name != self._selected_mesh:
                    self._selected_mesh = name
                    self.update()
                    self.selection_changed.emit(name)
            else:
                # Click on empty space — orbit camera
                self._dragging = True

    def mouseMoveEvent(self, e):
        dx = e.pos().x() - self._last_pos.x()
        dy = e.pos().y() - self._last_pos.y()
        self._last_pos = e.pos()

        is_shift = bool(e.modifiers() & _SHIFT)

        if self._pose_dragging and self._selected_mesh:
            mesh = self.scene.get_mesh(self._selected_mesh)
            if mesh is not None:
                self._apply_pose_drag(mesh, dx, dy)
                self.update()
                self.pose_changed.emit(self._selected_mesh)
                self.camera_changed.emit()   # triggers auto-update
        elif self._dragging:
            if is_shift:
                self._do_pan(dx, dy)
            else:
                self.azimuth    += dx * 0.5
                self.elevation   = float(np.clip(self.elevation - dy * 0.5, -89, 89))
            self.update();  self.camera_changed.emit()
        elif self._pan_dragging:
            self._do_pan(dx, dy)
            self.update();  self.camera_changed.emit()

    def mouseReleaseEvent(self, e):
        self._dragging      = False
        self._pan_dragging  = False
        self._pose_dragging = False

    def wheelEvent(self, e):
        try:
            delta = e.angleDelta().y()
        except AttributeError:
            delta = e.delta()
        self.distance = max(0.0001, self.distance * (0.88 if delta > 0 else 1.0 / 0.88))
        self.update();  self.camera_changed.emit()

    # ── Helpers ───────────────────────────────────────────────────────────
    def _do_pan(self, dx, dy):
        right, up = self._camera_axes()
        s = self.distance * 0.001
        self.target -= right * (dx * s)
        self.target += up    * (dy * s)

    def _apply_pose_drag(self, mesh, dx, dy):
        if self.tool_mode == ToolMode.ROTATE:
            mesh.pose[1] += dx * 0.5   # Ry — horizontal drag
            mesh.pose[0] += dy * 0.5   # Rx — vertical drag
        elif self.tool_mode == ToolMode.MOVE:
            right, up = self._camera_axes()
            scale = self.distance * 0.002
            # Move in camera right/up plane
            delta = right * (dx * scale) - up * (dy * scale)
            mesh.pose[3] += delta[0]
            mesh.pose[4] += delta[1]
            mesh.pose[5] += delta[2]

    def _eye(self):
        az = np.radians(self.azimuth)
        el = np.radians(self.elevation)
        return self.target + self.distance * np.array(
            [np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)],
            dtype=np.float32)

    def _camera_axes(self):
        eye = self._eye()
        fwd = self.target - eye;  fwd /= np.linalg.norm(fwd)
        up  = np.array([0, 1, 0], dtype=np.float32)
        r   = np.cross(fwd, up);  r /= np.linalg.norm(r)
        u   = np.cross(r, fwd)
        return r, u

    def _ray_from_pixel(self, px: int, py: int):
        """Unproject screen pixel → (origin, direction) in world space."""
        w, h = self.width(), max(self.height(), 1)
        ndc_x = (2.0 * px / w) - 1.0
        ndc_y = 1.0 - (2.0 * py / h)

        proj = _perspective(self.fov, w / h, 0.001, 10000.0)
        eye  = self._eye()
        view = _lookat(eye, self.target, np.array([0, 1, 0], dtype=np.float32))

        inv_vp = np.linalg.inv(proj @ view)

        near_c = np.array([ndc_x, ndc_y, -1, 1], dtype=np.float64)
        far_c  = np.array([ndc_x, ndc_y,  1, 1], dtype=np.float64)

        near_w = inv_vp @ near_c;  near_w /= near_w[3]
        far_w  = inv_vp @ far_c;   far_w  /= far_w[3]

        origin    = near_w[:3].astype(np.float32)
        direction = (far_w[:3] - near_w[:3]).astype(np.float32)
        direction /= np.linalg.norm(direction)
        return origin, direction

    def _pick_mesh(self, px: int, py: int) -> str | None:
        """Ray-cast to find which mesh was clicked. Returns mesh name or None."""
        if not self.scene.meshes:
            return None
        try:
            origin, direction = self._ray_from_pixel(px, py)
        except Exception:
            return None

        nearest_t    = np.inf
        nearest_name = None

        for mesh in self.scene.meshes:
            try:
                T     = mesh.world_transform()
                inv_T = np.linalg.inv(T)

                # Transform ray to mesh-local space (don't normalise — preserves t)
                ro = (inv_T @ np.append(origin,    1.0).astype(np.float64))[:3].astype(np.float32)
                rd = (inv_T @ np.append(direction, 0.0).astype(np.float64))[:3].astype(np.float32)

                tris = mesh.indices.reshape(-1, 3)
                t    = _ray_mesh_hit(ro, rd, mesh.vertices, tris)

                if t < nearest_t:
                    nearest_t    = t
                    nearest_name = mesh.name
            except Exception:
                continue

        return nearest_name if nearest_t < np.inf else None

    def _rebuild_gpu_buffers(self):
        if not OPENGL_OK:
            return
        self.makeCurrent()
        for gpu in self._gpu.values():
            gpu.free()
        self._gpu.clear()
        for mesh in self.scene.meshes:
            gpu = _MeshGPU()
            try:
                gpu.upload(mesh.vertices, mesh.normals, mesh.indices)
                self._gpu[mesh.name] = gpu
            except Exception as exc:
                print(f"[krita3d] GPU upload failed for {mesh.name}: {exc}")
