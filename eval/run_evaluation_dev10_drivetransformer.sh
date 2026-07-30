#!/bin/bash
# Closed-loop eval on Dev10 routes (10 routes) with a single GPU.
# Matches the manuscript's dev10 protocol.
set -eu

B2D_ROOT=/home/wenjj/Documents/02-Research/10-OMOway/Bench2Drive
DT_ROOT=/home/wenjj/Documents/02-Research/10-OMOway/DriveTransformer

ALGO=DriveTransformer
PLANNER_TYPE=only_traj
GPU_RANK=${GPU_RANK:-0}

ROUTES=leaderboard/data/drivetransformer_bench2drive_dev10.xml
TEAM_AGENT=team_code/drivetransformer_b2d_agent.py
TEAM_CONFIG=DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py+DriveTransformer/ckpts/drivetransformer_large.pth

CHECKPOINT_ENDPOINT=DriveTransformer_b2d_${PLANNER_TYPE}/eval_dev10_full.json
SAVE_PATH=${B2D_ROOT}/eval_dev10_${ALGO}_${PLANNER_TYPE}

mkdir -p ${ALGO}_b2d_${PLANNER_TYPE}
mkdir -p ${SAVE_PATH}

# Resolve symlinks so PYTHONPATH points to the real DriveTransformer tree.
# Script lives in DriveTransformer/eval/, so DT_ROOT is one level up.
SCRIPT_PATH="${BASH_SOURCE[0]}"
while [ -L "$SCRIPT_PATH" ]; do SCRIPT_PATH="$(readlink -f "$SCRIPT_PATH")"; done
DT_ROOT_RESOLVED="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"

export PYTHONPATH=${DT_ROOT_RESOLVED}:${B2D_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export CARLA_ROOT=/home/wenjj/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=${PYTHONPATH}:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=${PYTHONPATH}:leaderboard:leaderboard/team_code:scenario_runner
export SCENARIO_RUNNER_ROOT=scenario_runner
export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export IS_BENCH2DRIVE=True
export PLANNER_TYPE=${PLANNER_TYPE}
export GPU_RANK=${GPU_RANK}
export SAVE_PATH=${SAVE_PATH}
export ROUTES=${ROUTES}
export TEAM_AGENT=${TEAM_AGENT}
export TEAM_CONFIG=${TEAM_CONFIG}
export CHECKPOINT_ENDPOINT=${CHECKPOINT_ENDPOINT}

echo "==== DriveTransformer dev10 (10 routes) ===="
echo "GPU_RANK:    ${GPU_RANK}"
echo "ROUTES:      ${ROUTES}"
echo "CHECKPOINT:  ${CHECKPOINT_ENDPOINT}"
echo "SAVE_PATH:   ${SAVE_PATH}"
echo "PYTHONPATH:  ${PYTHONPATH}"
echo "============================================"

cd "${B2D_ROOT}"

CUDA_VISIBLE_DEVICES=${GPU_RANK} python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
    --routes=${ROUTES} \
    --repetitions=1 \
    --track=${CHALLENGE_TRACK_CODENAME} \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --debug=0 \
    --resume=True \
    --port=30000 \
    --traffic-manager-port=50000 \
    --gpu-rank=${GPU_RANK}
