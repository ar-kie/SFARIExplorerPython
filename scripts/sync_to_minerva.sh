#!/bin/bash
# Copy this repository (code only) to Minerva.
#
# Preferred: clone from GitHub on Minerva once, then `git pull` to update:
#   cd /sc/arion/projects/ad-omics/raphael/SFARI && git clone https://github.com/ar-kie/SFARIExplorerPython SFARIExplorer
#
# This script is for pushing local, not-yet-committed changes directly:
#   bash scripts/sync_to_minerva.sh <minerva_user> [destination]
set -euo pipefail
USER_NAME="${1:?usage: sync_to_minerva.sh <minerva_user> [destination]}"
DEST="${2:-/sc/arion/projects/ad-omics/raphael/SFARI/SFARIExplorer}"
HOST="${MINERVA_HOST:-minerva.hpc.mssm.edu}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rsync -avz --exclude '.git/' --exclude-from "$REPO/.gitignore" "$REPO/" "${USER_NAME}@${HOST}:${DEST}/"
echo "Synced to ${HOST}:${DEST}"
