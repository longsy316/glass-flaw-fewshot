#!/bin/bash
# Like queue.sh, but every job waits until the GPU has at least NEED_MB free memory (shared machine).
# activate your Python environment before running this script
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
Q=${1:-jobs.txt}; NEED_MB=${NEED_MB:-18000}
while true; do
  next=$(while read -r out cmd; do [ -z "$out" ] && continue; [[ "$out" == \#* ]] && continue
         [ -f "$out/eval_final.json" ] || [ -f "$out/FAILED" ] || { echo "$out $cmd"; break; }; done < "$Q")
  [ -z "$next" ] && { echo "queue empty $(date)"; exit 0; }
  out=${next%% *}; cmd=${next#* }
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$NEED_MB" ]; do sleep 60; done
  mkdir -p "$out"; echo "[$(date '+%m-%d %H:%M')] START $out (free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) MiB)"
  eval "$cmd --out $out" > "$out/log.txt" 2>&1 || { touch "$out/FAILED"; echo "FAILED $out"; }
  echo "[$(date '+%m-%d %H:%M')] END $out $(grep -h '^final' $out/log.txt | tail -1 | cut -c1-120)"
done
