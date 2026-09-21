#!/bin/sh
# Opt-in launcher. Never changes or replaces the original CARLA executable.
set -eu
: "${CARLA_ORIGINAL_ROOT:?Set CARLA_ORIGINAL_ROOT to the original CARLA installation}"
binary="$CARLA_ORIGINAL_ROOT/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
expected=1da989e4ff136e0658fdea421bc1da2b0e5cb018c6e767efc4f6129c21ab1a81
actual=$(sha256sum "$binary")
[ "${actual%% *}" = "$expected" ] || { echo 'Refusing unmatched CARLA binary' >&2; exit 42; }
guard_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${GDB:-gdb}" -nx -batch -x "$guard_dir/guard.gdb" --args "$binary" CarlaUE4 "$@"
