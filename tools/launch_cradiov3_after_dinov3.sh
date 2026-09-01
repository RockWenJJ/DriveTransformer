#!/usr/bin/env bash
# Queues the C-RADIOv3-B backbone run behind the two DINOv3 runs.
#
# Deliberately a separate file from launch_dinov3_backbones_after_dgrcta_eval.sh:
# that script is already executing, and bash reads a running script lazily by
# byte offset, so editing it in place would corrupt its execution.
set -u

ROOT=/data5T/wenjj/Projects/DriveTransformer
PY=/data5T/wenjj/miniconda3/envs/drivetransformer/bin/python
QUEUE_LOG=/data5T/wenjj/tmp_scripts/cradiov3_backbone_queue.log
WEIGHTS=$ROOT/ckpts/c-radio_v3-b/model.safetensors
NAME=drivetransformer_base_cradiov3_b
CONFIG=adzoo/drivetransformer/configs/drivetransformer/drivetransformer_base_cradiov3_b.py
PORT=35633
IDLE_MEM_MIB=1000
IDLE_PASSES_NEEDED=5

DEPENDENCIES=(
  drivetransformer_base_dinov3_vits16
  drivetransformer_base_dinov3_vitb16
)

cd "$ROOT" || exit 1

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$QUEUE_LOG"
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

dependencies_complete() {
  local name
  for name in "${DEPENDENCIES[@]}"; do
    checkpoint_complete \
      "$ROOT/adzoo/drivetransformer/work_dirs/drivetransformer/$name/iter_25000.pth" \
      || return 1
  done
}

training_pids() {
  pgrep -f 'adzoo/drivetransformer/train[.]py' || true
}

gpus_idle() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    -i 0,1,2,3 | awk -v limit="$IDLE_MEM_MIB" \
    'BEGIN { ok=1 } $1 >= limit { ok=0 } END { exit ok ? 0 : 1 }'
}

if [ ! -s "$WEIGHTS" ]; then
  log "missing C-RADIOv3-B weights at $WEIGHTS; download them before queueing"
  exit 1
fi

if pgrep -f "train[.]py.*${NAME}" >/dev/null; then
  log "$NAME is already running; refusing a duplicate queue"
  exit 0
fi

WORK_DIR=$ROOT/adzoo/drivetransformer/work_dirs/drivetransformer/$NAME
CHECKPOINT=$WORK_DIR/iter_25000.pth
if checkpoint_complete "$CHECKPOINT"; then
  log "$NAME already has a valid iter_25000.pth; nothing to do"
  exit 0
fi
if find "$WORK_DIR" -maxdepth 1 -name 'iter_*.pth' -print -quit 2>/dev/null | grep -q .; then
  log "$NAME has a partial checkpoint; refusing an implicit overwrite/resume"
  exit 1
fi

log 'waiting for both DINOv3 runs to finish and GPU0-3 to go idle'
idle_passes=0
while true; do
  complete=no
  busy="$(training_pids | tr '\n' ' ')"
  dependencies_complete && complete=yes

  if [ "$complete" = yes ] && [ -z "$busy" ] && gpus_idle; then
    idle_passes=$((idle_passes + 1))
  else
    idle_passes=0
  fi

  log "dinov3_done=$complete train_pids=[$busy] idle_passes=$idle_passes/$IDLE_PASSES_NEEDED"
  [ "$idle_passes" -ge "$IDLE_PASSES_NEEDED" ] && break
  sleep 60
done

available_gb=$(df -BG --output=avail /data5T | tail -1 | tr -dc '0-9')
if [ "$available_gb" -lt 50 ]; then
  log "only ${available_gb} GiB free on /data5T; training not started"
  exit 1
fi

mkdir -p "$WORK_DIR"
TRAIN_LOG=/data5T/wenjj/tmp_scripts/${NAME}_train.log
: > "$TRAIN_LOG"
log "starting $NAME on GPU0-3 (batch=20 per GPU, iterations=25000, port=$PORT)"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_NAME="$NAME" \
  "$PY" -m torch.distributed.launch \
  --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="$PORT" \
  adzoo/drivetransformer/train.py "$CONFIG" \
  --work-dir "$WORK_DIR" --launcher pytorch \
  >> "$TRAIN_LOG" 2>&1
status=$?
log "$NAME exited with status $status"
[ "$status" -eq 0 ] || exit "$status"
checkpoint_complete "$CHECKPOINT" || {
  log "$NAME did not produce a valid iter_25000.pth"
  exit 1
}
log 'C-RADIOv3-B training completed successfully'
