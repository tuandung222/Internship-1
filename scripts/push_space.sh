#!/usr/bin/env bash
# Push the CoRGI Gradio demo to a Hugging Face Space.
#
# Requirements:
#   - `huggingface_hub` installed (`pip install huggingface_hub`)
#   - Logged in via `hf auth login` or provide HF_TOKEN env var.
#   - Git configured with your Hugging Face credentials.
#
# Usage:
#   HF_TOKEN=... ./scripts/push_space.sh
#
# Optional environment variables:
#   SPACE_ID           # default: tuandunghcmut/corgi-qwen3-vl-demo
#   SPACE_HARDWARE     # default: cpu-basic (ZeroGPU tier)
#   SPACE_TEMP_DIR     # default: ./_hf_space
#
# The script will clone (or initialize) the Space repo, sync project files,
# commit, and push the update. Rerun whenever you need to redeploy.

set -euo pipefail

SPACE_ID="${SPACE_ID:-tuandunghcmut/corgi-qwen3-vl-demo}"
SPACE_HARDWARE="${SPACE_HARDWARE:-cpu-basic}"
SPACE_TEMP_DIR="${SPACE_TEMP_DIR:-./_hf_space}"
export SPACE_ID SPACE_HARDWARE

HF_CLI="python -m huggingface_hub.cli.hf"
# Ensure the CLI module is importable.
if ! python -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "[!] huggingface_hub not installed. Install with 'pip install huggingface_hub'." >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "[!] rsync is required for file synchronization. Install rsync and retry." >&2
  exit 1
fi

# Authenticate if needed.
if ! ${HF_CLI} auth whoami >/dev/null 2>&1; then
  echo "[i] huggingface CLI not logged in; attempting login..."
  ${HF_CLI} auth login
fi

echo "[i] Ensuring Space ${SPACE_ID} exists (hardware=${SPACE_HARDWARE})."
python - <<'PY'
import os
from huggingface_hub import HfApi, SpaceHardware

space_id = os.environ["SPACE_ID"]
hardware = os.environ.get("SPACE_HARDWARE", "cpu-basic")
token = os.environ.get("HF_TOKEN")
token = token or None

api = HfApi(token=token)
api.create_repo(
    repo_id=space_id,
    repo_type="space",
    space_sdk="gradio",
    exist_ok=True,
)

# Update hardware only if requested (skip when hardware is empty).
if hardware and hardware != "cpu-basic":
    try:
        hardware_enum = SpaceHardware(hardware)
    except ValueError:
        print(f"[w] Unknown hardware '{hardware}', skipping hardware request.")
    else:
        try:
            api.request_space_hardware(space_id, hardware_enum, token=token)
        except Exception as exc:  # noqa: BLE001
            print(f"[w] Unable to set hardware to {hardware}: {exc}")
PY

SPACE_URL="https://huggingface.co/spaces/${SPACE_ID}"

rm -rf "${SPACE_TEMP_DIR}"
git clone "${SPACE_URL}" "${SPACE_TEMP_DIR}"

sync_paths=(
  app.py
  requirements.txt
  corgi
  examples
  README.md
  PROJECT_PLAN.md
  PROGRESS_LOG.md
  QWEN_INFERENCE_NOTES.md
)

for path in "${sync_paths[@]}"; do
  if [ -e "${path}" ]; then
    rsync -a --delete "${path}" "${SPACE_TEMP_DIR}/"
  fi
done

pushd "${SPACE_TEMP_DIR}" >/dev/null

git add .
if git diff --cached --quiet; then
  echo "[i] No changes to push."
else
  git commit -m "Deploy latest CoRGI Gradio demo"
  git push origin main
fi

popd >/dev/null

echo "[✓] Deployment complete: ${SPACE_URL}"
