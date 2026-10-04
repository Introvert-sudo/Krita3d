#!/usr/bin/env bash
# Install the Krita 3D Reference plugin
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYKRITA_DIR="$HOME/.local/share/krita/pykrita"

echo "=== Installing dependencies ==="
sudo pacman -S --needed --noconfirm python-trimesh python-numpy python-opengl python-pillow

echo ""
echo "=== Installing plugin ==="
mkdir -p "$PYKRITA_DIR"

# Copy desktop file
cp "$SCRIPT_DIR/krita3d.desktop" "$PYKRITA_DIR/"

# Copy plugin directory (overwrite)
rm -rf "$PYKRITA_DIR/krita3d"
cp -r "$SCRIPT_DIR/krita3d" "$PYKRITA_DIR/"

echo ""
echo "✓ Plugin installed to $PYKRITA_DIR"
echo ""
echo "Next steps:"
echo "  1. Open Krita"
echo "  2. Settings → Configure Krita → Python Plugin Manager"
echo "  3. Enable 'Krita 3D Reference' and click OK"
echo "  4. Restart Krita"
echo "  5. The '3D Reference' docker will appear in the View → Dockers menu"
