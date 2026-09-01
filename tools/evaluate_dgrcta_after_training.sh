#!/usr/bin/env bash
set -u

DT=/data5T/wenjj/Projects/DriveTransformer
B2D=/data5T/wenjj/Projects/Bench2Drive
CARLA=/data5T/wenjj/carla
PY=/data5T/wenjj/miniconda3/envs/drivetransformer/bin/python
CFG=/home/wenjj/Projects/DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_base_dgrcta.py
CKPT=/home/wenjj/Projects/DriveTransformer/adzoo/drivetransformer/work_dirs/drivetransformer/drivetransformer_base_dgrcta/iter_25000.pth
ROUTES=$B2D/leaderboard/data/bench2drive_dev94.xml
RUN_EVAL=$B2D/eval_scripts/run_eval_dt.sh
OUT=$B2D/eval_results/dt_dgrcta_dev94/results.json
OPEN_OUT=$DT/ol_eval_results/tasks_dgrcta.json
BASE_OUT=$B2D/eval_results/dt_base_10ep_dev94_canonical/results.json
CROSS_OUT=$B2D/eval_results/dt_crosstemp_dev94_canonical/results.json
SUMMARY=$DT/ol_eval_results/dgrcta_comparison_summary.json
LOG=/data5T/wenjj/tmp_scripts/dgrcta_posteval.log
OPEN_LOG=/data5T/wenjj/tmp_scripts/dgrcta_openloop.log
IDLE_MEM_MIB=1000
IDLE_PASSES_NEEDED=5

cd "$DT" || exit 1

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

fail() {
  log "ERROR: $*"
  exit 1
}

training_pids() {
  pgrep -f 'train[.]py.*drivetransformer_base_dgrcta' || true
}

gpus_idle() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    -i 0,1,2,3 | awk -v limit="$IDLE_MEM_MIB" \
    'BEGIN { ok=1 } $1 >= limit { ok=0 } END { exit ok ? 0 : 1 }'
}

result_complete() {
  "$PY" - "$1" "$2" <<'PY'
import json
import sys

try:
    with open(sys.argv[1]) as handle:
        checkpoint = json.load(handle)['_checkpoint']
    completed, total = checkpoint['progress']
    expected = int(sys.argv[2])
    valid = completed == total == expected and len(checkpoint['records']) == expected
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

open_result_complete() {
  "$PY" - "$1" <<'PY'
import json
import sys

try:
    with open(sys.argv[1]) as handle:
        result = json.load(handle)
    metrics = result['metrics']
    valid = result['n_samples'] == 12806 and all(
        any(key.startswith(prefix) for key in metrics)
        for prefix in ('det/', 'motion/', 'map/', 'plan/'))
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

cleanup_port() {
  local rpc=$1
  local pids
  pids=$(pgrep -f "carla-rpc-port=${rpc}( |$)" || true)
  [ -z "$pids" ] && return
  kill $pids 2>/dev/null || true
  sleep 10
  pids=$(pgrep -f "carla-rpc-port=${rpc}( |$)" || true)
  [ -z "$pids" ] || kill -9 $pids 2>/dev/null || true
}

run_shard() {
  local gpu=$1
  local shard=$2
  local expected=$3
  local rpc=$4
  local tm_port=$5
  local run="dgrcta_94_c${shard}"
  local xml="$B2D/leaderboard/data/E94_c${shard}.xml"
  local result="$B2D/eval_results/$run/results.json"
  local shard_log="/data5T/wenjj/tmp_scripts/${run}.log"

  for attempt in $(seq 1 20); do
    if result_complete "$result" "$expected"; then
      log "$run complete"
      return 0
    fi
    cleanup_port "$rpc"
    log "$run attempt=$attempt gpu=$gpu rpc=$rpc"
    RPCPORT="$rpc" "$RUN_EVAL" \
      "$CFG" "$CKPT" "$gpu" "$xml" "$run" "$tm_port" \
      >> "$shard_log" 2>&1
    status=$?
    cleanup_port "$rpc"
    log "$run attempt=$attempt exited status=$status"
    sleep 20
  done
  log "$run failed after 20 attempts"
  return 1
}

if pgrep -f '^bash tools/evaluate_dgrcta_after_training[.]sh$' \
    | grep -vw "$$" >/dev/null; then
  fail 'another DGR-CTA post-training evaluator is already running'
fi

log 'waiting for completed DGR-CTA training and iter_25000.pth'
while [ ! -s "$CKPT" ] || [ -n "$(training_pids)" ]; do
  latest=$(find "$(dirname "$CKPT")" -maxdepth 1 -name 'iter_*.pth' \
    -printf '%f\n' 2>/dev/null | sort -V | tail -1)
  log "training_pids=[$(training_pids | tr '\n' ' ')] latest_checkpoint=${latest:-none}"
  sleep 300
done

log 'checking final checkpoint integrity'
"$PY" -c \
  "import torch; d=torch.load('$CKPT', map_location='cpu'); assert d.get('state_dict'); print('checkpoint tensors:', len(d['state_dict']))" \
  >> "$LOG" 2>&1 || fail 'iter_25000.pth could not be loaded'

idle_passes=0
while [ "$idle_passes" -lt "$IDLE_PASSES_NEEDED" ]; do
  if [ -z "$(training_pids)" ] && gpus_idle; then
    idle_passes=$((idle_passes + 1))
  else
    idle_passes=0
  fi
  log "post-training idle_passes=$idle_passes/$IDLE_PASSES_NEEDED"
  [ "$idle_passes" -ge "$IDLE_PASSES_NEEDED" ] || sleep 60
done

log 'starting four-way dev94 closed-loop evaluation on GPU 0-3'
run_shard 0 0 24 30900 50900 & p0=$!
run_shard 1 1 24 30910 50910 & p1=$!
run_shard 2 2 23 30920 50920 & p2=$!
run_shard 3 3 23 30930 50930 & p3=$!
failed=0
for pid in "$p0" "$p1" "$p2" "$p3"; do
  wait "$pid" || failed=1
done
[ "$failed" -eq 0 ] || fail 'one or more dev94 shards did not complete'

log 'merging dev94 shards with canonical leaderboard statistics'
export PYTHONPATH="$DT:$B2D:$CARLA/PythonAPI:$CARLA/PythonAPI/carla:$B2D/leaderboard:$B2D/scenario_runner"
"$PY" "$DT/tools/merge_dev94_results.py" \
  --inputs \
    "$B2D/eval_results/dgrcta_94_c0/results.json" \
    "$B2D/eval_results/dgrcta_94_c1/results.json" \
    "$B2D/eval_results/dgrcta_94_c2/results.json" \
    "$B2D/eval_results/dgrcta_94_c3/results.json" \
  --routes "$ROUTES" --out "$OUT" >> "$LOG" 2>&1 \
  || fail 'dev94 result merge failed'
result_complete "$OUT" 94 || fail 'merged dev94 result is incomplete'

log 'recomputing canonical baseline and CrossTemp references'
"$PY" "$DT/tools/merge_dev94_results.py" \
  --inputs \
    "$B2D/eval_results/b10ep_94_c0/results.json" \
    "$B2D/eval_results/b10ep_94_c1/results.json" \
    "$B2D/eval_results/b10ep_94_c2/results.json" \
    "$B2D/eval_results/b10ep_94_c3/results.json" \
  --routes "$ROUTES" --out "$BASE_OUT" >> "$LOG" 2>&1 \
  || fail 'canonical baseline merge failed'
"$PY" "$DT/tools/merge_dev94_results.py" \
  --inputs \
    "$B2D/eval_results/ct_94_c0/results.json" \
    "$B2D/eval_results/ct_94_c1/results.json" \
    "$B2D/eval_results/ct_94_c2/results.json" \
    "$B2D/eval_results/ct_94_c3/results.json" \
  --routes "$ROUTES" --out "$CROSS_OUT" >> "$LOG" 2>&1 \
  || fail 'canonical CrossTemp merge failed'

log 'computing Bench2Drive driving abilities'
ability_port=31990
cleanup_port "$ability_port"
export CARLA_ROOT="$CARLA"
CUDA_VISIBLE_DEVICES=0 "$PY" "$B2D/tools/ability_benchmark_gpu.py" \
  --file "$ROUTES" --result_file "$OUT" --port "$ability_port" --gpu 0 \
  >> "$LOG" 2>&1
ability_status=$?
cleanup_port "$ability_port"
[ "$ability_status" -eq 0 ] || fail 'Bench2Drive ability scoring failed'
[ -s "${OUT%.json}_ability.json" ] || fail 'ability result was not created'

log 'starting open-loop detection, prediction, mapping, and planning evaluation'
for attempt in 1 2 3; do
  if open_result_complete "$OPEN_OUT"; then
    break
  fi
  log "open-loop attempt=$attempt gpu=0"
  CUDA_VISIBLE_DEVICES=0 "$PY" /data5T/wenjj/tmp_scripts/ol_eval_tasks.py \
    --config "$CFG" --ckpt "$CKPT" --out "$OPEN_OUT" \
    > "$OPEN_LOG" 2>&1
  status=$?
  log "open-loop attempt=$attempt exited status=$status"
done
open_result_complete "$OPEN_OUT" || fail 'open-loop result is incomplete after three attempts'

"$PY" - \
  "$BASE_OUT" "$B2D/eval_results/dt_base_10ep_dev94/results_ability.json" \
  "$DT/ol_eval_results/tasks_base10ep.json" \
  "$CROSS_OUT" "$B2D/eval_results/dt_crosstemp_dev94/results_ability.json" \
  "$DT/ol_eval_results/tasks_crosstemp.json" \
  "$OUT" "${OUT%.json}_ability.json" "$OPEN_OUT" "$SUMMARY" <<'PY'
import json
import sys

def load_method(closed_path, ability_path, open_path):
    with open(closed_path) as handle:
        closed = json.load(handle)['_checkpoint']['global_record']
    with open(ability_path) as handle:
        ability = json.load(handle)
    with open(open_path) as handle:
        open_loop = json.load(handle)
    return {
        'checkpoint': open_loop['ckpt'],
        'closed_loop_dev94': closed,
        'bench2drive_abilities': ability,
        'open_loop': open_loop['metrics'],
    }

paths = sys.argv[1:-1]
out_path = sys.argv[-1]
summary = {
    'baseline': load_method(*paths[0:3]),
    'crosstemp': load_method(*paths[3:6]),
    'dgrcta': load_method(*paths[6:9]),
}
with open(out_path, 'w') as handle:
    json.dump(summary, handle, indent=2)
print(json.dumps(summary, indent=2))
PY

log "all evaluations complete; summary=$SUMMARY"
