#!/usr/bin/env bash
set -u

ROOT=/data5T/wenjj/Projects/DriveTransformer
PY=/data5T/wenjj/miniconda3/envs/drivetransformer/bin/python
CFG=adzoo/drivetransformer/configs/drivetransformer/drivetransformer_base_dgrcta.py
WD=adzoo/drivetransformer/work_dirs/drivetransformer/drivetransformer_base_dgrcta
LOG=/data5T/wenjj/tmp_scripts/dgrcta_after_eval.log
TRAIN_LOG=/data5T/wenjj/tmp_scripts/dgrcta_train.log
PORT=35617
IDLE_PASSES_NEEDED=5
IDLE_MEM_MIB=1000

cd "$ROOT" || exit 1

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

closed_loop_done() {
  "$PY" - \
    /data5T/wenjj/Projects/Bench2Drive/eval_results/b10ep_94_c0/results.json \
    /data5T/wenjj/Projects/Bench2Drive/eval_results/b10ep_94_c1/results.json \
    /data5T/wenjj/Projects/Bench2Drive/eval_results/b10ep_94_c2/results.json \
    /data5T/wenjj/Projects/Bench2Drive/eval_results/b10ep_94_c3/results.json \
    /data5T/wenjj/Projects/Bench2Drive/eval_results/ct_94_c0/results.json \
    /data5T/wenjj/Projects/Bench2Drive/eval_results/ct_94_c1/results.json \
    /data5T/wenjj/Projects/Bench2Drive/eval_results/ct_94_c2/results.json \
    /data5T/wenjj/Projects/Bench2Drive/eval_results/ct_94_c3/results.json <<'PY'
import json
import sys

for path in sys.argv[1:]:
    try:
        with open(path) as handle:
            checkpoint = json.load(handle)['_checkpoint']
        completed, total = checkpoint['progress']
        if completed < total or len(checkpoint.get('records', ())) < total:
            raise ValueError(f'incomplete progress {completed}/{total}')
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise SystemExit(1)
PY
}

open_loop_done() {
  "$PY" - \
    "$ROOT/ol_eval_results/tasks_base10ep.json" \
    "$ROOT/ol_eval_results/tasks_crosstemp.json" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    try:
        with open(path) as handle:
            result = json.load(handle)
        if not result:
            raise ValueError('empty evaluation result')
    except (OSError, ValueError, json.JSONDecodeError):
        raise SystemExit(1)
PY
}

evaluation_pids() {
  pgrep -f 'leaderboard_[e]valuator' || true
  pgrep -f 'CarlaUE4-Linux-[S]hipping' || true
  pgrep -f 'ol_eval_tasks[.]py' || true
}

gpus_idle() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    -i 0,1,2,3 | awk -v limit="$IDLE_MEM_MIB" \
    'BEGIN { ok=1 } $1 >= limit { ok=0 } END { exit ok ? 0 : 1 }'
}

if pgrep -f 'train[.]py.*drivetransformer_base_dgrcta' >/dev/null; then
  log 'DGR-CTA training is already running; refusing a duplicate launch'
  exit 0
fi

log 'waiting for all baseline and cross-temporal evaluations to finish'
idle_passes=0
while true; do
  closed=no
  open=no
  busy="$(evaluation_pids | tr '\n' ' ')"
  closed_loop_done && closed=yes
  open_loop_done && open=yes

  if [ "$closed" = yes ] && [ "$open" = yes ] && [ -z "$busy" ] && gpus_idle; then
    idle_passes=$((idle_passes + 1))
  else
    idle_passes=0
  fi

  log "closed_loop=$closed open_loop=$open eval_pids=[$busy] idle_passes=$idle_passes/$IDLE_PASSES_NEEDED"
  [ "$idle_passes" -ge "$IDLE_PASSES_NEEDED" ] && break
  sleep 60
done

if pgrep -f 'train[.]py.*drivetransformer_base_dgrcta' >/dev/null; then
  log 'DGR-CTA training appeared while waiting; refusing a duplicate launch'
  exit 0
fi

if find "$WD" -maxdepth 1 -name '*.pth' -print -quit 2>/dev/null | grep -q .; then
  log "existing checkpoint found in $WD; refusing to overwrite or resume implicitly"
  exit 1
fi

available_gb=$(df -BG --output=avail /data5T | tail -1 | tr -dc '0-9')
if [ "$available_gb" -lt 50 ]; then
  log "only ${available_gb} GiB free on /data5T; training not started"
  exit 1
fi

mkdir -p "$WD"
log "evaluations complete and GPU 0-3 stable; starting DGR-CTA training (batch=20, iterations=25000, port=$PORT)"
: > "$TRAIN_LOG"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_NAME=dt_base_dgrcta_10ep \
  "$PY" -m torch.distributed.launch \
  --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="$PORT" \
  adzoo/drivetransformer/train.py "$CFG" \
  --work-dir "$WD" --launcher pytorch \
  --cfg-options data.samples_per_gpu=20 runner.max_iters=25000 \
    checkpoint_config.interval=500 checkpoint_config.max_keep_ckpts=3 \
  >> "$TRAIN_LOG" 2>&1
status=$?

log "DGR-CTA training exited with status $status"
find "$WD" -maxdepth 1 -name '*.pth' -printf '%TY-%Tm-%Td %TH:%TM %p\n' \
  2>/dev/null | sort -r | head -3 | tee -a "$LOG"
exit "$status"
