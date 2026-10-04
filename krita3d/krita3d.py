"""Plugin entry point: registers the 3D Reference Docker."""

from krita import DockWidgetFactory, DockWidgetFactoryBase
from .docker import Krita3DDocker

Application.addDockWidgetFactory(
    DockWidgetFactory(
        "krita3d_docker",
        DockWidgetFactoryBase.DockPosition.DockRight,
        Krita3DDocker,
    )
)
