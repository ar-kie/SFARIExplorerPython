#!/bin/bash
# Full rebuild as chained LSF jobs: steps 01-05 (build) -> 06 (CONCORD, CPU) -> post-processing
# (pseudobulk, metadata, ages, within-species correction, parquets).
#
#   export CONDA_ENV=/path/to/concord R_ENV=/path/to/r_correction   # if not reachable by name
#   bash jobs/run_all.sh                                              # from the repository root
#
# Steps 01-05 skip themselves when checkpoints exist, and step 03 caches every dataset aligned to the
# gene list of the previous build. Adding datasets changes that list, so start from an empty
# pipeline_output/ (move the old one aside; this script refuses to run otherwise).
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export SFARI_ROOT="${SFARI_ROOT:-/sc/arion/projects/ad-omics/raphael/SFARI}"
CKPT="$SFARI_ROOT/pipeline_output/checkpoints"
if ls "$CKPT"/*.done >/dev/null 2>&1; then
  echo "Checkpoints from an earlier build exist in $CKPT, so steps 01-05 would be skipped." >&2
  echo "Move the old output aside first:  mv $SFARI_ROOT/pipeline_output $SFARI_ROOT/pipeline_output_<date>" >&2
  exit 1
fi

OUT=$(bash jobs/run_pipeline_01-05.sh)
echo "$OUT"
JOB5=$(echo "$OUT" | sed -n 's/^Step 5: Job //p')
[ -n "$JOB5" ] || { echo "Could not read the job id of step 05." >&2; exit 1; }

JOB6=$(bsub -w "done($JOB5)" < jobs/run_integrate_concord.lsf | grep -oE '[0-9]+' | head -1)
echo "Step 06 (CONCORD): Job $JOB6"
JOB7=$(bsub -w "done($JOB6)" < jobs/run_postprocess.lsf | grep -oE '[0-9]+' | head -1)
echo "Step 07 (post-processing): Job $JOB7"
echo "Monitor with: bjobs -w"
