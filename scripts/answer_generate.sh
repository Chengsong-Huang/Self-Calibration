#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <MODEL_PATH> <OUTPUT_DIR> <GENERATION_COUNT>"
  exit 1
fi

MODEL_PATH="$1"
OUTPUT_DIR="$2"
GENERATION_COUNT=$3
TASKS=(
  # "object_counting"
  # "gsm8k"
  "math_qa"
  # "arc_challenge"
  # "arc_easy"
  # "svamp"
)
TEMPERATURE=1.0
NGPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "Number of GPUs: $NGPUS"


for TASK in "${TASKS[@]}"; do
  OUTPUT_FILE="${OUTPUT_DIR}/${TASK}.json"
  if [ -f "$OUTPUT_FILE" ]; then
    echo "Output file for task $TASK already exists. Skipping..."
    continue
  fi
  echo "Generating responses for task: $TASK"
  bash scripts/generate_response.sh -m "$MODEL_PATH" -d "$TASK" -o "$OUTPUT_FILE" -g $GENERATION_COUNT -t $TEMPERATURE -s $NGPUS
done

