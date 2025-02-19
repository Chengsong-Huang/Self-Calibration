#!/bin/bash

#bash main.bash ./models/llama llama meta-llama/Llama-3.1-8B-Instruct

# Check for required arguments
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <merged_model_path> <version> <basemodel>"
  exit 1
fi

MERGED_MODEL_PATH=$1
VERSION=$2
BASEMODEL=$3
# Step 1: Train model
torchrun --nproc_per_node=4 --master_port=12345 model_training/train.py \
  --config_file model_training/configs/${VERSION}.json \
  --save_path "./loras/${VERSION}"

# Step 2: Merge LoRA model
python model_training/merge_lora_model.py \
  --base_model_name ${BASEMODEL} \
  --lora_path "./loras/${VERSION}" \
  --merged_model_path ${MERGED_MODEL_PATH}



