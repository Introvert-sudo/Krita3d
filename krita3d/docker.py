"""Main Krita Docker: wraps the 3D viewport, pose controls and layer push."""

from __future__ import annotations
import os

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
        QPushButton, QCheckBox, QLabel, QSlider, QTabWidget,
        QFileDialog, QMessageBox, QSizePolicy, QToolButton, QButtonGroup,
    )
    from PyQt6.QtCore import Qt, QTimer  # noqa: F401
    _Qt_Horizontal   = Qt.Orientation.Horizontal
    _Qt_Vertical     = Qt.Orientation.Vertical
    _SP_Ignored      = QSizePolicy.Policy.Ignored
    _SP_Preferred    = QSizePolicy.Policy.Preferred
except ImportError:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
        QPushButton, QCheckBox, QLabel, QSlider, QTabWidget,
        QFileDialog, QMessageBox, QSizePolicy, QToolButton, QButtonGroup,
    )
    from PyQt5.QtCore import Qt, QTimer
    _Qt_Horizontal   = Qt.Horizontal
    _Qt_Vertical     = Qt.Vertical
    _SP_Ignored      = QSizePolicy.Ignored
    _SP_Preferred    = QSizePolicy.Preferred

from krita import DockWidget

from .scene    import Scene3D, TRIMESH_OK
from .viewport import Viewport3D, OPENGL_OK, ToolMode
from .pose_widget import PoseWidget
from .renderer import push_to_krita


_SUPPORTED = "3D Models (*.obj *.glb *.gltf *.stl *.ply *.off)"


class Krita3DDocker(DockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Reference")

        self._scene = Scene3D()
        self._auto_update = False
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._update_layer)

        # ── Root widget ───────────────────────────────────────────────────
        root = QWidget()
        self.setWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        # ── Top toolbar ───────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        self._btn_open = QPushButton("Open 3D File…")
        self._btn_open.clicked.connect(self._open_file)
        self._lbl_file = QLabel("No file loaded")
        self._lbl_file.setSizePolicy(_SP_Ignored, _SP_Preferred)
        toolbar.addWidget(self._btn_open)
        toolbar.addWidget(self._lbl_file, 1)
        vbox.addLayout(toolbar)

        # ── Warnings ──────────────────────────────────────────────────────
        if not TRIMESH_OK:
            w = QLabel("⚠ trimesh not found — install python-trimesh")
            w.setStyleSheet("color: orange;")
            vbox.addWidget(w)
        if not OPENGL_OK:
            w = QLabel("⚠ PyOpenGL not found — install python-opengl")
            w.setStyleSheet("color: orange;")
            vbox.addWidget(w)

        # ── Tool buttons ──────────────────────────────────────────────────
        tool_bar = QHBoxLayout()
        tool_bar.setSpacing(2)
        self._tool_group = QButtonGroup(root)
        self._tool_group.setExclusive(True)
        for label, mode, tip in [
            ("Orbit",  ToolMode.ORBIT,  "Orbit / pan camera (Shift+drag to pan)"),
            ("Rotate", ToolMode.ROTATE, "Click mesh to select, drag to rotate"),
            ("Move",   ToolMode.MOVE,   "Click mesh to select, drag to move"),
        ]:
            btn = QToolButton()
            btn.setText(label)
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.setProperty("tool_mode", mode)
            self._tool_group.addButton(btn)
            tool_bar.addWidget(btn)
        tool_bar.addStretch()
        self._tool_group.buttons()[0].setChecked(True)   # Orbit default
        self._tool_group.buttonClicked.connect(self._on_tool_clicked)
        vbox.addLayout(tool_bar)

        # ── Splitter: viewport | controls ────────────────────────────────
        splitter = QSplitter(_Qt_Vertical)
        vbox.addWidget(splitter, 1)

        # Viewport
        self._viewport = Viewport3D(self._scene)
        self._viewport.setMinimumHeight(120)
        self._viewport.camera_changed.connect(self._on_camera_changed)
        self._viewport.selection_changed.connect(self._on_selection_changed)
        self._viewport.pose_changed.connect(self._on_viewport_pose_changed)
        splitter.addWidget(self._viewport)

        # Controls panel (tabs)
        ctrl = QWidget()
        ctrl_vbox = QVBoxLayout(ctrl)
        ctrl_vbox.setContentsMargins(0, 0, 0, 0)
        self._tabs = QTabWidget()
        ctrl_vbox.addWidget(self._tabs)
        tabs = self._tabs
        splitter.addWidget(ctrl)
        splitter.setSizes([300, 200])

        # Tab 1 — Camera
        cam_tab = QWidget()
        cam_layout = QVBoxLayout(cam_tab)
        cam_layout.setContentsMargins(4, 4, 4, 4)
        cam_layout.setSpacing(4)

        # FOV slider
        fov_row = QHBoxLayout()
        fov_row.addWidget(QLabel("FOV:"))
        self._fov_slider = QSlider(_Qt_Horizontal)
        self._fov_slider.setRange(10, 120)
        self._fov_slider.setValue(45)
        self._fov_label = QLabel("45°")
        self._fov_slider.valueChanged.connect(self._on_fov_changed)
        fov_row.addWidget(self._fov_slider, 1)
        fov_row.addWidget(self._fov_label)
        cam_layout.addLayout(fov_row)

        # Wireframe
        self._chk_wire = QCheckBox("Wireframe")
        self._chk_wire.toggled.connect(self._viewport.set_wireframe)
        cam_layout.addWidget(self._chk_wire)

        # Reset view
        btn_reset = QPushButton("Reset View")
        btn_reset.clicked.connect(self._viewport.reset_view)
        cam_layout.addWidget(btn_reset)
        cam_layout.addStretch()
        tabs.addTab(cam_tab, "Camera")

        # Tab 2 — Pose
        self._pose_widget = PoseWidget(self._scene)
        self._pose_widget.pose_changed.connect(self._on_pose_changed)
        tabs.addTab(self._pose_widget, "Pose")

        # ── Bottom bar ────────────────────────────────────────────────────
        bottom = QHBoxLayout()
        self._btn_update = QPushButton("▶ Update Layer")
        self._btn_update.clicked.connect(self._update_layer)
        bottom.addWidget(self._btn_update, 1)
        self._chk_auto = QCheckBox("Auto")
        self._chk_auto.setToolTip("Update layer automatically when camera or pose changes")
        self._chk_auto.toggled.connect(self._set_auto)
        bottom.addWidget(self._chk_auto)
        vbox.addLayout(bottom)

    # ── canvasChanged is mandatory in DockWidget ──────────────────────────
    def canvasChanged(self, canvas):
        pass

    # ── Slots ─────────────────────────────────────────────────────────────
    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open 3D Model", "", _SUPPORTED)
        if not path:
            return
        try:
            count = self._scene.load(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        fname = os.path.basename(path)
        self._lbl_file.setText(f"{fname}  ({count} mesh{'es' if count != 1 else ''})")
        self._viewport.set_scene(self._scene)
        self._pose_widget.set_scene(self._scene)

    def _on_fov_changed(self, val: int):
        self._fov_label.setText(f"{val}°")
        self._viewport.fov = float(val)
        self._viewport.update()
        self._trigger_auto()

    def _on_pose_changed(self, *_):
        self._viewport.rebuild()
        self._trigger_auto()

    def _on_tool_clicked(self, btn):
        mode = btn.property("tool_mode")
        self._viewport.set_tool_mode(mode)
        if mode != ToolMode.ORBIT:
            self._tabs.setCurrentIndex(1)   # show Pose tab

    def _on_selection_changed(self, name: str):
        self._pose_widget.set_selected(name)
        self._tabs.setCurrentIndex(1)   # switch to Pose tab

    def _on_viewport_pose_changed(self, name: str):
        self._pose_widget.sync_pose(name)
        self._trigger_auto()

    def _on_camera_changed(self):
        self._trigger_auto()

    def _set_auto(self, on: bool):
        self._auto_update = on

    def _trigger_auto(self):
        if self._auto_update:
            self._debounce.start()

    def _update_layer(self):
        try:
            from krita import Krita
            doc = Krita.instance().activeDocument()
        except Exception:
            doc = None
        if doc is None:
            QMessageBox.warning(self, "No Document", "Open a Krita document first.")
            return
        ok = push_to_krita(self._viewport, doc)
        if not ok:
            QMessageBox.warning(self, "Error", "Could not push render to layer.")
