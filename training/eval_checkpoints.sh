#!/usr/bin/env bash
# Run eval_offline.py on every checkpoint of a training run as soon as it is saved; results go to outputs/eval/<run>/.
# Usage: training/eval_checkpoints.sh <run-name> [arguments for eval_offline.py]
# Stops after the checkpoint of the run's last step. Checkpoints with an existing report are skipped.
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="${1:?Укажи имя запуска из outputs/train}"
shift
RUN_DIR="$PROJECT/outputs/train/$RUN"
OUT="$PROJECT/outputs/eval/$RUN"
mkdir -p "$OUT"

while true; do
    newest="$(ls "$RUN_DIR/checkpoints" 2>/dev/null | grep -E '^[0-9]+$' | sort | tail -1 || true)"
    for dir in "$RUN_DIR"/checkpoints/[0-9]*; do
        name="$(basename "$dir")"
        [ -d "$dir/pretrained_model" ] && [ ! -f "$OUT/$name.txt" ] || continue
        # training_state is written after the weights, so its presence means the checkpoint is (nearly) complete;
        # older checkpoints lose it once a newer one is saved (lerobot_train_fast.py).
        [ -d "$dir/training_state" ] || [ "$name" != "$newest" ] || continue
        sleep 30
        echo "$(date '+%F %T') чекпойнт $name"
        python "$PROJECT/training/eval_offline.py" --policy "$dir/pretrained_model" --output "$OUT/$name.json" "$@" \
            > "$OUT/$name.txt" 2>&1 || echo "  ошибка, см. $OUT/$name.txt"
        grep -E "^Ошибка с правильной" "$OUT/$name.txt" || true
    done
    last="$(ls "$RUN_DIR/checkpoints" 2>/dev/null | grep -E '^[0-9]+$' | sort | tail -1 || true)"
    if [ -n "$last" ] && [ -f "$OUT/$last.txt" ] && python3 -c "import json, sys; sys.exit(json.load(open(sys.argv[1]))['steps'] != int(sys.argv[2]))" \
        "$RUN_DIR/checkpoints/$last/pretrained_model/train_config.json" "$last"; then
        break
    fi
    sleep 60
done
