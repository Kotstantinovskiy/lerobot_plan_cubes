#!/usr/bin/env bash
# Download the 20-episode pilot archive from Google Drive, check its SHA-256 and unpack it into
# data/color_cubes_wrist1080. The Drive folder must be shared as "anyone with the link".
# Parts already in WORK are reused, so an interrupted download can be finished by running this again.
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
FOLDER_URL="${FOLDER_URL:-https://drive.google.com/drive/folders/1D4aSdV4vm7ryPv3zdcS6Fi2IeXNFuZvW}"
WORK="${WORK:-$PROJECT/data/downloads/pilot20}"
NAME=color_cubes_wrist1080_pilot20.zip
PARTS=32
SHA256=92dea7f290bdb5d8185e18ccaa32e742a959b638aacf4181a22b812a6e7235ca
TARGET="$PROJECT/data/color_cubes_wrist1080"

if [ -e "$TARGET" ]; then
    echo "Датасет уже есть: $TARGET" >&2
    exit 0
fi
mkdir -p "$WORK"
count() { find "$WORK" -maxdepth 1 -name "$NAME.part-*" | wc -l | tr -d ' '; }
if [ "$(count)" -ne "$PARTS" ]; then
    command -v gdown >/dev/null || { echo "Нужен gdown (pip install gdown или training/setup_linux.sh)" >&2; exit 1; }
    gdown --folder "$FOLDER_URL" -O "$WORK" --continue
fi
[ "$(count)" -eq "$PARTS" ] || { echo "Ожидалось $PARTS частей, в $WORK их $(count)" >&2; exit 1; }

ZIP="$WORK/$NAME"
cat "$WORK/$NAME".part-* > "$ZIP"
if command -v sha256sum >/dev/null; then
    echo "$SHA256  $ZIP" | sha256sum -c -
else
    echo "$SHA256  $ZIP" | shasum -a 256 -c -
fi
unzip -q "$ZIP" 'color_cubes_wrist1080/*' -d "$PROJECT/data"
rm "$ZIP"
echo "Готово: $TARGET ($(python3 -c "import json,sys; i=json.load(open(sys.argv[1])); print(i['total_episodes'], 'эпизодов,', i['total_frames'], 'кадров')" "$TARGET/meta/info.json"))"
