#!/usr/bin/env bash

model_name="$1"
generate_output_folder="$2"
confidence_output_folder="$3"
if [ $# -lt 3 ]; then
  echo "Error: Missing arguments."
  echo "Usage: bash run_confidence.sh <model_name> <generate_output_folder> <confidence_output_folder>"
  exit 1
fi

datasets=(
  "object_counting"
  "gsm8k"
  "math_qa"
  "arc_challenge"
  "arc_easy"
  "svamp"
)

GPU_QUEUE=($(nvidia-smi --query-gpu=index --format=csv,noheader))
echo "Available GPUs: ${GPU_QUEUE[@]}"

declare -A pids

DATASET_INDEX=0
NUM_DATASETS=${#datasets[@]}

start_job() {
  local gpu_id="$1"
  local dataset="$2"

  echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Start dataset [${dataset}] on GPU [${gpu_id}] ..."

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  python evaluation/calculate_confidence.py \
    --responses_file "${generate_output_folder}/${dataset}.json" \
    --model_name "${model_name}" \
    --output_folder "${confidence_output_folder}/${dataset}/" \
    --dataset_name "${dataset}" \
    --model_type "${model_name}" & 

  pids["${gpu_id}"]=$!
}

while :; do
  while [ ${#GPU_QUEUE[@]} -gt 0 ] && [ ${DATASET_INDEX} -lt ${NUM_DATASETS} ]; do
    gpu_id="${GPU_QUEUE[0]}"
    GPU_QUEUE=("${GPU_QUEUE[@]:1}")

    dataset="${datasets[${DATASET_INDEX}]}"
    ((DATASET_INDEX++))

    start_job "$gpu_id" "$dataset"
  done

  if [ ${DATASET_INDEX} -ge ${NUM_DATASETS} ] && [ ${#pids[@]} -eq 0 ]; then
    break
  fi

  for gpu_id in "${!pids[@]}"; do
    pid="${pids[$gpu_id]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] job finished with PID [${pid}]."
      unset pids["$gpu_id"]
      GPU_QUEUE+=("$gpu_id")
    fi
  done

  sleep 1
done

echo "==> All tasks have finished!"
