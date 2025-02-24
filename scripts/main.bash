#!/bin/bash

MERGED_MODEL_PATH="./models/llama"
VERSION="llama"
BASEMODEL="meta-llama/Llama-3.1-8B-Instruct"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --merged_model_path) MERGED_MODEL_PATH="$2"; shift 2;;
    --version) VERSION="$2"; shift 2;;
    --basemodel) BASEMODEL="$2"; shift 2;;
    -h|--help)
      echo "Usage: bash main.bash [--merged_model_path <path>] [--version <version>] [--basemodel <base>]"
      exit 0;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

echo "Running with:"
echo "  Merged Model Path: $MERGED_MODEL_PATH"
echo "  Version          : $VERSION"
echo "  Base Model       : $BASEMODEL"
# Step 1: Train model
torchrun --nproc_per_node=4 --master_port=12345 model_training/train.py \
  --config_file model_training/configs/${VERSION}.json \
  --save_path "./loras/${VERSION}"

# Step 2: Merge LoRA model
python model_training/merge_lora_model.py \
  --base_model_name ${BASEMODEL} \
  --lora_path "./loras/${VERSION}" \
  --merged_model_path ${MERGED_MODEL_PATH}



