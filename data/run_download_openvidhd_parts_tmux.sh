#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANK_RANGE="${1:-1-10}"
SESSION_NAME="${2:-openvidhd_${RANK_RANGE//-/_}}"

if [[ $# -ge 1 ]]; then
  shift
fi
if [[ $# -ge 1 ]]; then
  shift
fi

EXTRA_ARGS=("$@")
LOG_ROOT="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_ROOT"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
CONSOLE_LOG="${LOG_ROOT}/tmux_${SESSION_NAME}_${RUN_TS}.log"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  echo "Use: tmux attach -t $SESSION_NAME" >&2
  exit 1
fi

CMD=(python3 download_openvidhd_parts.py --rank-range "$RANK_RANGE")
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

printf -v QUOTED_CMD '%q ' "${CMD[@]}"
printf -v QUOTED_SCRIPT_DIR '%q' "$SCRIPT_DIR"
printf -v QUOTED_LOG '%q' "$CONSOLE_LOG"

tmux new-session -d -s "$SESSION_NAME" "cd ${QUOTED_SCRIPT_DIR} && ${QUOTED_CMD} 2>&1 | tee -a ${QUOTED_LOG}"

echo "Started tmux session: $SESSION_NAME"
echo "Rank range: $RANK_RANGE"
echo "Console log: $CONSOLE_LOG"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Watch log: tail -f $CONSOLE_LOG"