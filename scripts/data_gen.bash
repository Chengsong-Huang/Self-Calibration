#!/bin/bash

# Default values
model_name="meta-llama/Llama-3.1-8B-Instruct"
temperature=0.8
use_cot_flag="--use_cot"
num_generations=32
subset="train"
data_size=100
save_path="llama"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name) model_name="$2"; shift 2;;
    --temperature) temperature="$2"; shift 2;;
    --use_cot_flag) use_cot_flag="$2"; shift 2;;
    --num_generations) num_generations="$2"; shift 2;;
    --subset) subset="$2"; shift 2;;
    --data_size) data_size="$2"; shift 2;;
    --save_path) save_path="$2"; shift 2;;
    -h|--help)
      echo "Usage: bash data_gen.bash [--model_name <model>] [--temperature <temp>] [--use_cot_flag <flag>] [--num_generations <num>] [--subset <subset>] [--data_size <size>] [--save_path <path>]"
      exit 0;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

echo "Running with:"
echo "  Model Name     : $model_name"
echo "  Temperature    : $temperature"
echo "  Use COT Flag   : $use_cot_flag"
echo "  Num Generations: $num_generations"
echo "  Subset         : $subset"
echo "  Data Size      : $data_size"
echo "  Save Path      : $save_path"

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
