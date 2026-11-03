#!/usr/bin/env bash
# Download a dataset archive from Google Drive, check its SHA-256 and unpack it into data/color_cubes_wrist1080.
#   bash training/fetch_dataset.sh [ep108|ep88|ep68|ep48|ep40|pilot20]   (default ep108)
# The Drive folder must be shared as "anyone with the link". gdown lists at most 50 files per folder, so the
# parts of ep88 and ep68 sit in three subfolders and those of ep48 and ep40 in two; parts are found recursively and
# joined in name order.
# ep108 holds only what episodes 88-107 added to ep88 (their data and video files and the whole meta/): ep88 is
# fetched first, or kept if it is already in place, and ep108 is unpacked over it.
# A different dataset already in the target (e.g. the 20-episode pilot) is moved to data/color_cubes_wrist1080_<N>ep.
# Parts already in WORK are reused, so an interrupted download can be finished by running this again.
set -euo pipefail

PRESET="${1:-ep108}"
BASE=
case "$PRESET" in
    ep108)
        DEFAULT_URL=https://drive.google.com/drive/folders/1qEmUvTFyYnP3BfyLZwE1oKjqXpCh7NtN
        NAME=color_cubes_wrist1080_108ep_from88.zip
        PARTS=31
        EPISODES=108
        BASE=ep88
        SHA256=b1f6085ce53aaae4a18a987947c03177fd9c50fed10f393385082b6e61038d8b
        ;;
    ep88)
        DEFAULT_URL=https://drive.google.com/drive/folders/179750bQ4Kb8wnSu_vHNq8HCdMWc6PfSo
        NAME=color_cubes_wrist1080_88ep.zip
        PARTS=133
        EPISODES=88
        SHA256=ab006970dc693df535f3525a4fe7f322bc16d043f9102e163f1758bd9e7b8a02
        ;;
    ep68)
        DEFAULT_URL=https://drive.google.com/drive/folders/1ZoyJ0-L1_R_XfA_66XokQmRRgJSwq0Lo
        NAME=color_cubes_wrist1080_68ep.zip
        PARTS=106
        EPISODES=68
        SHA256=98c5d97825391390a6eb22759bdadfb92cecef8ce33d01c4eaba8524b545927c
        ;;
    ep48)
        DEFAULT_URL=https://drive.google.com/drive/folders/1CKQFklWspc674p4F4v-mjotlabGUON-3
        NAME=color_cubes_wrist1080_48ep.zip
        PARTS=73
        EPISODES=48
        SHA256=196db3cef0e01e8e1593c00583eb723b8d6fd2572eb8faac26e43f5c94075033
        ;;
    ep40)
        DEFAULT_URL=https://drive.google.com/drive/folders/17Dh6FnDNPUPyLdmQSvEOm5X-lgMPtiX3
        NAME=color_cubes_wrist1080_40ep.zip
        PARTS=62
        EPISODES=40
        SHA256=a7f8b43d30f27b169c225bb24c6325b825cc58d7d7b57906053959498a96423e
        ;;
    pilot20)
        DEFAULT_URL=https://drive.google.com/drive/folders/1D4aSdV4vm7ryPv3zdcS6Fi2IeXNFuZvW
        NAME=color_cubes_wrist1080_pilot20.zip
        PARTS=32
        EPISODES=20
        SHA256=92dea7f290bdb5d8185e18ccaa32e742a959b638aacf4181a22b812a6e7235ca
        ;;
    *) echo "Неизвестный набор: $PRESET (есть ep108, ep88, ep68, ep48, ep40, pilot20)" >&2; exit 2 ;;
esac

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
FOLDER_URL="${FOLDER_URL:-$DEFAULT_URL}"
WORK="${WORK:-$PROJECT/data/downloads/$PRESET}"
TARGET="$PROJECT/data/color_cubes_wrist1080"
episodes() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['total_episodes'])" "$1/meta/info.json"; }

if [ -e "$TARGET" ] && [ "$(episodes "$TARGET")" -eq "$EPISODES" ]; then
    echo "Датасет уже есть: $TARGET ($EPISODES эпизодов)" >&2
    exit 0
fi
if [ -n "$BASE" ]; then
    # The base with its own folder and download directory, whatever FOLDER_URL and WORK say for this archive.
    env -u FOLDER_URL -u WORK bash "$0" "$BASE"
elif [ -e "$TARGET" ]; then
    HAVE="$(episodes "$TARGET")"
    ASIDE="${TARGET}_${HAVE}ep"
    [ ! -e "$ASIDE" ] || { echo "Не могу отложить $TARGET: $ASIDE уже существует" >&2; exit 1; }
    mv "$TARGET" "$ASIDE"
    echo "Прежний датасет ($HAVE эпизодов) перенесён в $ASIDE"
fi
mkdir -p "$WORK"
parts() { find "$WORK" -name "$NAME.part-*" | awk -F/ '{print $NF "\t" $0}' | sort | cut -f2; }
if [ "$(parts | wc -l | tr -d ' ')" -ne "$PARTS" ]; then
    command -v gdown >/dev/null || { echo "Нужен gdown (pip install gdown или training/setup_linux.sh)" >&2; exit 1; }
    gdown --folder "$FOLDER_URL" -O "$WORK" --continue
fi
COUNT="$(parts | wc -l | tr -d ' ')"
[ "$COUNT" -eq "$PARTS" ] || { echo "Ожидалось $PARTS частей, в $WORK их $COUNT" >&2; exit 1; }

ZIP="$WORK/$NAME"
parts | while IFS= read -r part; do cat "$part"; done > "$ZIP"
if command -v sha256sum >/dev/null; then
    echo "$SHA256  $ZIP" | sha256sum -c -
else
    echo "$SHA256  $ZIP" | shasum -a 256 -c -
fi
unzip -q -o "$ZIP" 'color_cubes_wrist1080/*' -d "$PROJECT/data"
rm "$ZIP"
HAVE="$(episodes "$TARGET")"
[ "$HAVE" -eq "$EPISODES" ] || { echo "В $TARGET после распаковки $HAVE эпизодов, а не $EPISODES" >&2; exit 1; }
echo "Готово: $TARGET ($(python3 -c "import json,sys; i=json.load(open(sys.argv[1])); print(i['total_episodes'], 'эпизодов,', i['total_frames'], 'кадров')" "$TARGET/meta/info.json"))"
