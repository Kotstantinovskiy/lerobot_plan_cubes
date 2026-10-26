#!/bin/sh
# Launch from any working directory without changing the existing LeRobot setup.
set -eu

app_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
hardware_mode=false
for argument in "$@"; do
    if [ "$argument" = "--hardware" ]; then
        hardware_mode=true
    fi
done

if [ "$hardware_mode" = true ]; then
    robot_python="$app_dir/.venv-robot/bin/python"
    if [ ! -x "$robot_python" ]; then
        printf '%s\n' \
            "Hardware environment is missing: $app_dir/.venv-robot" \
            "Create it from your existing LeRobot Python and install requirements.txt." \
            "See $app_dir/README.md for the two setup commands." >&2
        exit 1
    fi
    exec "$robot_python" "$app_dir/app.py" "$@"
fi

if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' "Install uv, then run this launcher again. See $app_dir/README.md." >&2
    exit 1
fi
exec uv run --project "$app_dir" python "$app_dir/app.py" "$@"
