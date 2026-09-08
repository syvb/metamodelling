#!/usr/bin/env bash
# Run a command under a hard cgroup memory cap so a runaway process gets killed
# instead of thrashing the whole VM. Usage: MEM=10G scripts/memrun.sh cmd args...
exec systemd-run --user --scope -q -p MemoryMax="${MEM:-10G}" -p MemorySwapMax="${SWAPMAX:-1G}" --collect "$@"
