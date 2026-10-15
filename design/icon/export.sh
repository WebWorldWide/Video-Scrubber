#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Rebuilds AppIcon.icon and exports the app icons from it. Needs Xcode 26+ (actool and Icon Composer).
#
#   sh design/icon/export.sh
#
# Outputs:
#   src-tauri/icons/macos/Assets.car   Liquid Glass icon for macOS 26+ (CFBundleIconName)
#   src-tauri/icons/icon.icns          fallback for earlier macOS (CFBundleIconFile), 16–1024 px
#   src-tauri/icons/*.png, icon.ico    Windows and Linux, rendered from the same document
#   design/icon/preview.png            1024 px render for the README
set -eu

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ICON="$ROOT/design/icon/AppIcon.icon"
ICTOOL="$(xcode-select -p)/../Applications/Icon Composer.app/Contents/Executables/ictool"
TMP=$(mktemp -d)
trap 'rm -rf "${TMP:?}"' EXIT

python3 "$ROOT/design/icon/build_icon.py"

xcrun actool "$ICON" --compile "$TMP" --platform macosx --minimum-deployment-target 11.0 \
  --app-icon AppIcon --output-partial-info-plist "$TMP/partial.plist" >/dev/null

# Render at 1024 and place it on the macOS icon grid (824 px body, centred) for the .icns,
# Windows and Linux icons; actool's own fallback .icns stops at 256 px.
"$ICTOOL" "$ICON" --export-image --output-file "$TMP/full.png" \
  --platform macOS --rendition Default --width 824 --height 824 --scale 1 >/dev/null
sips --padToHeightWidth 1024 1024 "$TMP/full.png" --out "$TMP/grid.png" >/dev/null
(cd "$ROOT" && npx tauri icon "$TMP/grid.png" -o src-tauri/icons >/dev/null)
rm -rf "${ROOT:?}/src-tauri/icons/android" "${ROOT:?}/src-tauri/icons/ios"

mkdir -p "$ROOT/src-tauri/icons/macos"
cp "$TMP/Assets.car" "$ROOT/src-tauri/icons/macos/Assets.car"

"$ICTOOL" "$ICON" --export-image --output-file "$ROOT/design/icon/preview.png" \
  --platform macOS --rendition Default --width 1024 --height 1024 --scale 1 >/dev/null
echo "Exported icons from $ICON"
