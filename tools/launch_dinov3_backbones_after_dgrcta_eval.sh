#!/usr/bin/env bash
set -u

ROOT=/data5T/wenjj/Projects/DriveTransformer
PY=/data5T/wenjj/miniconda3/envs/drivetransformer/bin/python
SUMMARY=$ROOT/ol_eval_results/dgrcta_comparison_summary.json
QUEUE_LOG=/data5T/wenjj/tmp_scripts/dinov3_backbone_queue.log
IDLE_MEM_MIB=1000
IDLE_PASSES_NEEDED=5

cd "$ROOT" || exit 1

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$QUEUE_LOG"
}

dgrcta_evaluation_complete() {
  "$PY" - "$SUMMARY" <<'PY'
import json
import sys

try:
    with open(sys.argv[1]) as handle:
        summary = json.load(handle)
    result = summary["dgrcta"]
    closed = result["closed_loop_dev94"]
    abilities = result["bench2drive_abilities"]
    metrics = result["open_loop"]
    valid = (
        bool(closed)
        and bool(abilities)
        and all(
            any(key.startswith(prefix) for key in metrics)
            for prefix in ("det/", "motion/", "map/", "plan/")
        )
    )
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

evaluation_pids() {
  pgrep -f 'leaderboard_[e]valuator' || true
  pgrep -f 'CarlaUE4-Linux-[S]hipping' || true
  pgrep -f 'ol_eval_tasks[.]py' || true
  pgrep -f 'ability_benchmark_[g]pu.py' || true
}

gpus_idle() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    -i 0,1,2,3 | awk -v limit="$IDLE_MEM_MIB" \
    'BEGIN { ok=1 } $1 >= limit { ok=0 } END { exit ok ? 0 : 1 }'
}

checkpoint_complete() {
  local checkpoint=$1
  [ -s "$checkpoint" ] || return 1
  "$PY" - "$checkpoint" <<'PY'
import sys
import torch

try:
    checkpoint = torch.load(sys.argv[1], map_location="cpu")
    valid = bool(checkpoint.get("state_dict"))
except (OSError, RuntimeError, KeyError, TypeError, ValueError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

run_training() {
  local name=$1
  local config=$2
  local port=$3
  local work_dir=$ROOT/adzoo/drivetransformer/work_dirs/drivetransformer/$name
  local checkpoint=$work_dir/iter_25000.pth
  local train_log=/data5T/wenjj/tmp_scripts/${name}_train.log

  if checkpoint_complete "$checkpoint"; then
    log "$name already has a valid iter_25000.pth; skipping"
    return 0
  fi
  if pgrep -f "train[.]py.*${config}" >/dev/null; then
    log "$name is already running; refusing a duplicate launch"
    return 1
  fi
  if find "$work_dir" -maxdepth 1 -name 'iter_*.pth' -print -quit \
      2>/dev/null | grep -q .; then
    log "$name has a partial checkpoint; refusing an implicit overwrite/resume"
    return 1
  fi

  mkdir -p "$work_dir"
  : > "$train_log"
  log "starting $name on GPU0-3 (batch=20 per GPU, iterations=25000, port=$port)"
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  WANDB_NAME="$name" \
    "$PY" -m torch.distributed.launch \
    --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="$port" \
    adzoo/drivetransformer/train.py "$config" \
    --work-dir "$work_dir" --launcher pytorch \
    >> "$train_log" 2>&1
  local status=$?
  log "$name exited with status $status"
  [ "$status" -eq 0 ] || return "$status"
  checkpoint_complete "$checkpoint" || {
    log "$name did not produce a valid iter_25000.pth"
    return 1
  }
}

if pgrep -f 'train[.]py.*drivetransformer_base_dinov3_' >/dev/null; then
  log 'a DINOv3 training process is already running; refusing a duplicate queue'
  exit 0
fi

log 'waiting for complete DGR-CTA closed-loop, ability, and open-loop results'
idle_passes=0
while true; do
  complete=no
  busy="$(evaluation_pids | tr '\n' ' ')"
  dgrcta_evaluation_complete && complete=yes

  if [ "$complete" = yes ] && [ -z "$busy" ] && gpus_idle; then
    idle_passes=$((idle_passes + 1))
  else
    idle_passes=0
  fi

  log "dgrcta_eval=$complete eval_pids=[$busy] idle_passes=$idle_passes/$IDLE_PASSES_NEEDED"
  [ "$idle_passes" -ge "$IDLE_PASSES_NEEDED" ] && break
  sleep 60
done

available_gb=$(df -BG --output=avail /data5T | tail -1 | tr -dc '0-9')
if [ "$available_gb" -lt 50 ]; then
  log "only ${available_gb} GiB free on /data5T; training not started"
  exit 1
fi

run_training \
  drivetransformer_base_dinov3_vits16 \
  adzoo/drivetransformer/configs/drivetransformer/drivetransformer_base_dinov3_vits16.py \
  35631 || exit $?

run_training \
  drivetransformer_base_dinov3_vitb16 \
  adzoo/drivetransformer/configs/drivetransformer/drivetransformer_base_dinov3_vitb16.py \
  35632 || exit $?

log 'ViT-S/16 and ViT-B/16 training completed successfully'

