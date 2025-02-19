#!/usr/bin/env bash




model_name="$1"
temperature="$2"
use_cot_flag="$3"
num_generations="$4"
subset="$5"
data_size="$6"
save_path="$7"

if [ $# -lt 7 ]; then
  echo "Error: Missing arguments."
  echo "Usage: bash run_generation.sh <model_name> <temperature> <use_cot_flag> <num_generations> <subset> <data_size> <save_path>"
  exit 1
fi

datasets=(
    "gsm8k"
    # "sciq"
    # "commonsense_qa"
    # "winogrande"
    # "openbookqa"
    # "reclor"
    # "arc_easy"
    # "logiqa"
    # "svamp"
)

GPU_QUEUE=($(nvidia-smi --query-gpu=index --format=csv,noheader))
echo "==> Available GPUs: ${GPU_QUEUE[@]}"

declare -A pids

DATASET_INDEX=0
NUM_DATASETS=${#datasets[@]}

start_job() {
  local gpu_id="$1"
  local dataset="$2"

  echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Start dataset [${dataset}] on GPU [${gpu_id}] ..."

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  # python data_creation/data_generator.py \
  #     --model_name "${model_name}" \
  #     --temperature "${temperature}" \
  #     ${use_cot_flag} \
  #     --dataset_name "${dataset}" \
  #     --num_generations "${num_generations}" \
  #     --subset "${subset}" \
  #     --data_size "${data_size}" \
  #     --n_GPU 1 \
  #     --save_path "${save_path}" &

  python data_creation/data_generator_dt.py \
      --model_name "${model_name}" \
      --temperature "${temperature}" \
      ${use_cot_flag} \
      --dataset_name "${dataset}" \
      --num_generations "${num_generations}" \
      --subset "${subset}" \
      --data_size "${data_size}" \
      --n_GPU 1 \
      --save_path "${save_path}_dt" &

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
