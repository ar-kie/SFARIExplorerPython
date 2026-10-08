#!/bin/bash
# Download the app's parquet build from the Hugging Face Space into app/data/.
# The files are too large for GitHub (temporal_mean.parquet > 100 MB), so the Space is
# the canonical copy of the deployed build.
#   bash scripts/fetch_app_data.sh
set -euo pipefail
SPACE="${SPACE:-ar-kie/SFARIExplorer}"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/app/data"
FILES="celltype_meta dataset_info dataset_overview expression_mean expression_meta expression_pct
       gene_map risk_genes stage_mapping summary_statistics temporal_mean temporal_meta umap_subsample"
mkdir -p "$DEST"
for f in $FILES; do
  if [ -s "$DEST/$f.parquet" ]; then echo "exists: $f.parquet"; continue; fi
  echo "downloading $f.parquet"
  curl -fL --retry 3 -o "$DEST/$f.parquet.part" "https://huggingface.co/spaces/${SPACE}/resolve/main/data/${f}.parquet"
  mv "$DEST/$f.parquet.part" "$DEST/$f.parquet"
done
for f in build_info.json dataset_references.json; do   # optional files
  curl -fsL -o "$DEST/$f" "https://huggingface.co/spaces/${SPACE}/resolve/main/data/$f" 2>/dev/null || rm -f "$DEST/$f"
done
echo "App data in $DEST"
