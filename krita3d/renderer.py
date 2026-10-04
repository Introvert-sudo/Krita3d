"""Push the viewport framebuffer into a Krita paint layer."""

try:
    from PyQt6.QtGui import QImage, QPainter
    from PyQt6.QtCore import Qt
    _FMT_ARGB32 = QImage.Format.Format_ARGB32
    _SMOOTH = Qt.TransformationMode.SmoothTransformation
    _KEEP   = Qt.AspectRatioMode.KeepAspectRatio
except ImportError:
    from PyQt5.QtGui import QImage, QPainter
    from PyQt5.QtCore import Qt
    _FMT_ARGB32 = QImage.Format_ARGB32
    _SMOOTH = Qt.SmoothTransformation
    _KEEP   = Qt.KeepAspectRatio

_LAYER_NAME = "3D Reference"


def _find_or_create_layer(doc):
    """Return existing '3D Reference' paintlayer or create a new one."""
    root = doc.rootNode()
    for child in root.childNodes():
        if child.name() == _LAYER_NAME and child.type() == "paintlayer":
            return child
    node = doc.createNode(_LAYER_NAME, "paintlayer")
    root.addChildNode(node, None)
    return node


def _qimage_to_bytes(img: QImage) -> bytes:
    ptr = img.bits()
    try:
        n = img.sizeInBytes()
    except AttributeError:
        n = img.byteCount()
    ptr.setsize(n)   # sip.voidptr uses lowercase setsize in both PyQt5 and PyQt6
    return bytes(ptr)


def push_to_krita(viewport, doc=None):
    """
    Grab the viewport framebuffer and push it onto the '3D Reference' layer.

    Parameters
    ----------
    viewport : Viewport3D
    doc      : Krita Document — if None, uses the active document.

    Returns
    -------
    True on success, False if there is no active document.
    """
    if doc is None:
        try:
            from krita import Krita
            doc = Krita.instance().activeDocument()
        except Exception:
            return False

    if doc is None:
        return False

    dw, dh = doc.width(), doc.height()

    # Grab the OpenGL framebuffer
    viewport.makeCurrent()
    raw = viewport.grabFramebuffer()
    viewport.doneCurrent()

    if raw.isNull():
        return False

    # Scale to document size (letterboxed, centred)
    scaled = raw.scaled(dw, dh, _KEEP, _SMOOTH)
    scaled = scaled.convertToFormat(_FMT_ARGB32)

    canvas = QImage(dw, dh, _FMT_ARGB32)
    canvas.fill(0)
    painter = QPainter(canvas)
    x = (dw - scaled.width())  // 2
    y = (dh - scaled.height()) // 2
    painter.drawImage(x, y, scaled)
    painter.end()

    pixel_bytes = _qimage_to_bytes(canvas)
    node = _find_or_create_layer(doc)
    node.setPixelData(pixel_bytes, 0, 0, dw, dh)
    doc.refreshProjection()
    return True
