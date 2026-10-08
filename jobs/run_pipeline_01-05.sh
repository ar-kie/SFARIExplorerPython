#!/bin/bash
# Submit pipeline steps 01-05 (inspect, gene universe, concatenate, cell types, dev stages) as chained LSF jobs.
#   bash jobs/run_pipeline_01-05.sh            (from the repository root on Minerva)
set -e
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../pipeline" && pwd)"
SFARI_ROOT="${SFARI_ROOT:-/sc/arion/projects/ad-omics/raphael/SFARI}"
export SFARI_ROOT
OUTPUT_DIR="${SFARI_ROOT}/pipeline_output"
LOG_DIR="${OUTPUT_DIR}/logs"
CONDA_ENV="concord"
QUEUE="premium"; PROJECT="acc_ad-omics"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${OUTPUT_DIR}/checkpoints" "${OUTPUT_DIR}/temp"

echo "Submitting SFARI Pipeline V4..."

JOB1=$(bsub -q $QUEUE -P $PROJECT -n 1 -R "rusage[mem=350G]" -W 2:00 -J sfari_01 \
    -o ${LOG_DIR}/step01_%J.out -e ${LOG_DIR}/step01_%J.err \
    "source ~/.bashrc && conda activate $CONDA_ENV && cd $PIPELINE_DIR && python 01_inspect_datasets.py" | grep -oP '(?<=<)\d+(?=>)')
echo "Step 1: Job $JOB1"

JOB2=$(bsub -q $QUEUE -P $PROJECT -n 1 -R "rusage[mem=3650G]" -W 24:00 -J sfari_02 -w "done($JOB1)" \
    -o ${LOG_DIR}/step02_%J.out -e ${LOG_DIR}/step02_%J.err \
    "source ~/.bashrc && conda activate $CONDA_ENV && cd $PIPELINE_DIR && python 02_build_gene_universe.py" | grep -oP '(?<=<)\d+(?=>)')
echo "Step 2: Job $JOB2"

JOB3=$(bsub -q $QUEUE -P $PROJECT -n 1 -R "rusage[mem=1100G]" -W 144:00 -J sfari_03 -w "done($JOB2)" \
    -o ${LOG_DIR}/step03_%J.out -e ${LOG_DIR}/step03_%J.err \
    "source ~/.bashrc && conda activate $CONDA_ENV && cd $PIPELINE_DIR && python 03_concatenate_datasets.py" | grep -oP '(?<=<)\d+(?=>)')
echo "Step 3: Job $JOB3"

JOB4=$(bsub -q $QUEUE -P $PROJECT -n 1 -R "rusage[mem=1100G]" -W 72:00 -J sfari_04 -w "done($JOB3)" \
    -o ${LOG_DIR}/step04_%J.out -e ${LOG_DIR}/step04_%J.err \
    "source ~/.bashrc && conda activate $CONDA_ENV && cd $PIPELINE_DIR && python 04_annotate_celltypes.py" | grep -oP '(?<=<)\d+(?=>)')
echo "Step 4: Job $JOB4"

JOB5=$(bsub -q $QUEUE -P $PROJECT -n 1 -R "rusage[mem=1100G]" -W 72:00 -J sfari_05 -w "done($JOB4)" \
    -o ${LOG_DIR}/step05_%J.out -e ${LOG_DIR}/step05_%J.err \
    "source ~/.bashrc && conda activate $CONDA_ENV && cd $PIPELINE_DIR && python 05_annotate_devstage.py" | grep -oP '(?<=<)\d+(?=>)')
echo "Step 5: Job $JOB5"

echo -e "\nAll jobs submitted! Monitor: bjobs -w"
