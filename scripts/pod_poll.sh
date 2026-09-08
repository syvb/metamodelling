#!/usr/bin/env bash
# Poll pod status every 60s into a log until the pod disappears or 4h elapse.
LOG=${1:-data/pod_status.log}
for i in $(seq 1 240); do
  out=$(.venv/bin/python scripts/runpod_launch.py status 2>&1 | tail -3)
  echo "$(date +%H:%M:%S) $out" >> "$LOG"
  echo "$out" | grep -q delta-nla || { echo "$(date +%H:%M:%S) pod gone" >> "$LOG"; break; }
  sleep 60
done
