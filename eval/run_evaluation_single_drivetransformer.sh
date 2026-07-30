BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive220.xml
TEAM_AGENT=team_code/drivetransformer_b2d_agent.py
TEAM_CONFIG=DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py+DriveTransformer/ckpts/drivetransformer_large.pth
BASE_CHECKPOINT_ENDPOINT=eval_bench2drive220
ALGO=DriveTransformer
PLANNER_TYPE=only_traj
SAVE_PATH=./eval_bench2drive220_${ALGO}_${PLANNER_TYPE}

GPU_RANK=${GPU_RANK:-0}
TASK_ID=${TASK_ID:-0}

mkdir -p ${ALGO}_b2d_${PLANNER_TYPE}
mkdir -p ${SAVE_PATH}

PORT=${BASE_PORT}
TM_PORT=${BASE_TM_PORT}
ROUTES=${BASE_ROUTES}
CHECKPOINT_ENDPOINT="${ALGO}_b2d_${PLANNER_TYPE}/${BASE_CHECKPOINT_ENDPOINT}_full.json"

SCRIPT_PATH="${BASH_SOURCE[0]}"
while [ -L "$SCRIPT_PATH" ]; do SCRIPT_PATH="$(readlink -f "$SCRIPT_PATH")"; done
# Script lives in DriveTransformer/eval/, so DT_ROOT is one level up.
DT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
B2D_ROOT="$(cd "${DT_ROOT}/../Bench2Drive" 2>/dev/null && pwd)"
export PYTHONPATH=${DT_ROOT}${B2D_ROOT:+:${B2D_ROOT}}${PYTHONPATH:+:${PYTHONPATH}}:leaderboard:leaderboard/team_code:scenario_runner

echo "==== Single-GPU DriveTransformer bench2drive220 evaluation ===="
echo "GPU_RANK:        ${GPU_RANK}"
echo "ROUTES:          ${ROUTES}"
echo "CHECKPOINT:      ${CHECKPOINT_ENDPOINT}"
echo "SAVE_PATH:       ${SAVE_PATH}"
echo "TEAM_CONFIG:     ${TEAM_CONFIG}"
echo "PYTHONPATH:      ${PYTHONPATH}"
echo "============================================================"

bash -e leaderboard/scripts/run_evaluation.sh \
    ${PORT} ${TM_PORT} ${IS_BENCH2DRIVE} \
    ${ROUTES} ${TEAM_AGENT} ${TEAM_CONFIG} \
    ${CHECKPOINT_ENDPOINT} ${SAVE_PATH} \
    ${PLANNER_TYPE} ${GPU_RANK} \
    2>&1 | tee ${BASE_ROUTES%.*}_single.log
