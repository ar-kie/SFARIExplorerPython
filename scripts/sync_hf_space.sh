#!/bin/bash
# Copy the app code from this repository into the Hugging Face Space clone for deployment.
# The Space keeps its own data/ (Git LFS); this only touches code and config.
#   bash scripts/sync_hf_space.sh [path/to/hf_space]     (default: ../hf_space)
# Then review and push from the Space clone:  cd ../hf_space && git diff && git commit -am "..." && git push
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPACE_DIR="${1:-$REPO/../hf_space}"
[ -d "$SPACE_DIR/.git" ] || { echo "Not a git clone: $SPACE_DIR" >&2; exit 1; }
rsync -av --delete --exclude '__pycache__' "$REPO/app/sfx/" "$SPACE_DIR/sfx/"
rsync -av "$REPO/app/app.py" "$REPO/app/requirements.txt" "$REPO/app/README.md" "$SPACE_DIR/"
rsync -av "$REPO/app/.streamlit/" "$SPACE_DIR/.streamlit/"
git -C "$SPACE_DIR" status --short
