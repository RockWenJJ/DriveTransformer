#!/bin/bash
# Single-GPU bf16 smoke test: run N iters of DriveTransformer-Large training
# in bf16 mixed precision and report wall-clock time + it/s.
set -eu

DT_ROOT=/home/wenjj/Documents/02-Research/10-OMOway/DriveTransformer
CFG=adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py
WORK_DIR=${DT_ROOT}/work_dirs/bf16_smoke
MAX_ITERS=${MAX_ITERS:-100}
BATCH=${BATCH:-2}
WORKERS=${WORKERS:-4}
WARMUP_ITERS=${WARMUP_ITERS:-5}

mkdir -p "${WORK_DIR}"
cd "${DT_ROOT}"

export PYTHONPATH=${DT_ROOT}:${PYTHONPATH:-}
export WORK_DIR

# Skip dataset pre-filter warning by setting num_gpus=1 to match real setup
# Don't use the distributed launcher; single-process torch is enough.
CUDA_VISIBLE_DEVICES=0 /home/wenjj/miniconda3/envs/b2d_zoo/bin/python \
    adzoo/drivetransformer/train.py "${CFG}" \
    --work-dir "${WORK_DIR}" \
    --cfg-options \
        runner.max_iters="${MAX_ITERS}" \
        data.samples_per_gpu="${BATCH}" \
        data.workers_per_gpu="${WORKERS}" \
        log_config.interval=10 \
        checkpoint_config.interval=10000 \
    2>&1 | tee "${WORK_DIR}/smoke.$(date +%Y%m%d_%H%M%S).log"
