"""Pose controls: per-mesh transform sliders."""
from __future__ import annotations

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QDoubleSpinBox, QPushButton, QFormLayout, QGroupBox,
    )
    from PyQt6.QtCore import pyqtSignal, Qt
except ImportError:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QDoubleSpinBox, QPushButton, QFormLayout, QGroupBox,
    )
    from PyQt5.QtCore import pyqtSignal, Qt

from .scene import Scene3D


class PoseWidget(QWidget):
    """Shows a mesh selector and 6-DOF spinboxes (Rx Ry Rz Tx Ty Tz)."""

    pose_changed = pyqtSignal(str, float, float, float, float, float, float)

    _DOF = [
        ("Rx°", -360, 360, 1.0),
        ("Ry°", -360, 360, 1.0),
        ("Rz°", -360, 360, 1.0),
        ("Tx",  -100, 100, 0.01),
        ("Ty",  -100, 100, 0.01),
        ("Tz",  -100, 100, 0.01),
    ]

    def __init__(self, scene: Scene3D, parent=None):
        super().__init__(parent)
        self.scene = scene
        self._updating = False   # re-entrancy guard
        self._spins: list[QDoubleSpinBox] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Mesh selector
        h = QHBoxLayout()
        h.addWidget(QLabel("Mesh:"))
        self._combo = QComboBox()
        self._combo.setMinimumWidth(120)
        self._combo.currentIndexChanged.connect(self._on_mesh_selected)
        h.addWidget(self._combo, 1)
        layout.addLayout(h)

        # DOF spinboxes
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(3)
        for label, lo, hi, step in self._DOF:
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setDecimals(2)
            spin.valueChanged.connect(self._on_spin_changed)
            self._spins.append(spin)
            form.addRow(label, spin)
        layout.addLayout(form)

        # Reset button
        btn_reset = QPushButton("Reset Transform")
        btn_reset.clicked.connect(self._reset_current)
        layout.addWidget(btn_reset)
        layout.addStretch()

    # ── Public API ────────────────────────────────────────────────────────
    def set_scene(self, scene: Scene3D):
        self.scene = scene
        self._updating = True
        self._combo.clear()
        for name in scene.node_names():
            self._combo.addItem(name)
        self._updating = False
        self._load_current_pose()

    def current_mesh_name(self) -> str | None:
        return self._combo.currentText() or None

    def set_selected(self, name: str):
        """Select a mesh by name (called when viewport picks a mesh)."""
        idx = self._combo.findText(name)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)

    def sync_pose(self, name: str):
        """Reload spinboxes from scene data (called after viewport drag)."""
        if self.current_mesh_name() == name:
            self._load_current_pose()

    # ── Slots ─────────────────────────────────────────────────────────────
    def _on_mesh_selected(self, _idx):
        self._load_current_pose()

    def _on_spin_changed(self, _val):
        if self._updating:
            return
        name = self.current_mesh_name()
        if name is None:
            return
        mesh = self.scene.get_mesh(name)
        if mesh is None:
            return
        pose = [s.value() for s in self._spins]
        mesh.pose = pose
        self.pose_changed.emit(name, *pose)

    def _reset_current(self):
        name = self.current_mesh_name()
        if name is None:
            return
        mesh = self.scene.get_mesh(name)
        if mesh is None:
            return
        mesh.pose = [0.0] * 6
        self._load_current_pose()
        self.pose_changed.emit(name, 0, 0, 0, 0, 0, 0)

    def _load_current_pose(self):
        name = self.current_mesh_name()
        if name is None:
            return
        mesh = self.scene.get_mesh(name)
        if mesh is None:
            return
        self._updating = True
        for spin, val in zip(self._spins, mesh.pose):
            spin.setValue(val)
        self._updating = False
