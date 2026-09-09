#!/usr/bin/env bash
set -euo pipefail
DT_ROOT=${DT_ROOT:-/data5T/wenjj/Projects/DriveTransformer_worktrees/consistency_20260909/DriveTransformer}
DT_EVAL_PYTHON=${DT_EVAL_PYTHON:-/data5T/wenjj/miniconda3/envs/drivetransformer/bin/python}
exec "$DT_EVAL_PYTHON" "$DT_ROOT/tools/evaluate_dev24.py" "$@"
