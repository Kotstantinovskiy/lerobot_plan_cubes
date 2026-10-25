#!/usr/bin/env bash
# Linux + NVIDIA GPU: install the LeRobot fork with SmolVLA extras into <fork>/.venv for training.
# The fork sits next to this repository unless LEROBOT_DIR says otherwise.
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
LEROBOT_DIR="${LEROBOT_DIR:-$PROJECT/../lerobot_cubes}"
LEROBOT_REPO="${LEROBOT_REPO:-https://github.com/Kotstantinovskiy/lerobot_cubes.git}"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"

[ -x "$UV" ] || { echo "Нужен uv: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }

[ -d "$LEROBOT_DIR/.git" ] || git clone "$LEROBOT_REPO" "$LEROBOT_DIR"
cd "$LEROBOT_DIR"
[ -x .venv/bin/python ] || "$UV" venv --python 3.12 .venv
# gdown fetches the dataset archive from Google Drive (training/fetch_pilot.sh).
"$UV" pip install --python .venv/bin/python -e '.[smolvla]' gdown
.venv/bin/python -c "import torch, lerobot; print('lerobot', lerobot.__version__, '| torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
echo "Окружение: source $LEROBOT_DIR/.venv/bin/activate"
