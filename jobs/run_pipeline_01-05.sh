#!/bin/bash
# Submit pipeline steps 01-05 (inspect, gene universe, concatenate, cell types, dev stages) as chained LSF jobs.
#   bash jobs/run_pipeline_01-05.sh            (from the repository root on Minerva)
set -e
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../pipeline" && pwd)"
SFARI_ROOT="${SFARI_ROOT:-/sc/arion/projects/ad-omics/raphael/SFARI}"
export SFARI_ROOT
OUTPUT_DIR="${SFARI_ROOT}/pipeline_output"
LOG_DIR="${OUTPUT_DIR}/logs"
CONDA_ENV="${CONDA_ENV:-concord}"   # name or path of the conda env
QUEUE="premium"; PROJECT="acc_ad-omics"

# Memory (rusage is per core; each step runs on one core) and wall time. Override any of them, e.g.
#   MEM03=600G bash jobs/run_all.sh
# 01, 02  read gene names and 1,000 sampled cells per matrix, never whole matrices
# 03      one dataset at a time: ~2.5x its count matrix (the largest, ~1.5 M cells, needs ~150 GB),
#         then the combined obs while the matrices are written to disk one by one
# 04, 05  obs only; HDF5 copies the count matrix to the new file without loading it
MEM01="${MEM01:-64G}";  W01="${W01:-4:00}"
MEM02="${MEM02:-16G}";  W02="${W02:-1:00}"
MEM03="${MEM03:-400G}"; W03="${W03:-48:00}"
MEM04="${MEM04:-128G}"; W04="${W04:-24:00}"
MEM05="${MEM05:-128G}"; W05="${W05:-12:00}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${OUTPUT_DIR}/checkpoints" "${OUTPUT_DIR}/temp"

echo "Submitting SFARI Pipeline V4..."

JOB1=$(bsub -q $QUEUE -P $PROJECT -n 1 -R "rusage[mem=$MEM01]" -W $W01 -J sfari_01 \
    -o ${LOG_DIR}/step01_%J.out -e ${LOG_DIR}/step01_%J.err \
    "source ~/.bashrc && conda activate $CONDA_ENV && cd $PIPELINE_DIR && python 01_inspect_datasets.py" | grep -oP '(?<=<)\d+(?=>)')
echo "Step 1: Job $JOB1"

JOB2=$(bsub -q $QUEUE -P $PROJECT -n 1 -R "rusage[mem=$MEM02]" -W $W02 -J sfari_02 -w "done($JOB1)" \
    -o ${LOG_DIR}/step02_%J.out -e ${LOG_DIR}/step02_%J.err \
    "source ~/.bashrc && conda activate $CONDA_ENV && cd $PIPELINE_DIR && python 02_build_gene_universe.py" | grep -oP '(?<=<)\d+(?=>)')
echo "Step 2: Job $JOB2"

JOB3=$(bsub -q $QUEUE -P $PROJECT -n 1 -R "rusage[mem=$MEM03]" -W $W03 -J sfari_03 -w "done($JOB2)" \
    -o ${LOG_DIR}/step03_%J.out -e ${LOG_DIR}/step03_%J.err \
    "source ~/.bashrc && conda activate $CONDA_ENV && cd $PIPELINE_DIR && python 03_concatenate_datasets.py" | grep -oP '(?<=<)\d+(?=>)')
echo "Step 3: Job $JOB3"

JOB4=$(bsub -q $QUEUE -P $PROJECT -n 1 -R "rusage[mem=$MEM04]" -W $W04 -J sfari_04 -w "done($JOB3)" \
    -o ${LOG_DIR}/step04_%J.out -e ${LOG_DIR}/step04_%J.err \
    "source ~/.bashrc && conda activate $CONDA_ENV && cd $PIPELINE_DIR && python 04_annotate_celltypes.py" | grep -oP '(?<=<)\d+(?=>)')
echo "Step 4: Job $JOB4"

JOB5=$(bsub -q $QUEUE -P $PROJECT -n 1 -R "rusage[mem=$MEM05]" -W $W05 -J sfari_05 -w "done($JOB4)" \
    -o ${LOG_DIR}/step05_%J.out -e ${LOG_DIR}/step05_%J.err \
    "source ~/.bashrc && conda activate $CONDA_ENV && cd $PIPELINE_DIR && python 05_annotate_devstage.py" | grep -oP '(?<=<)\d+(?=>)')
echo "Step 5: Job $JOB5"

echo -e "\nAll jobs submitted! Monitor: bjobs -w"
