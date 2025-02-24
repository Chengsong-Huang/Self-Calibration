#!/bin/bash

MODEL="meta-llama/Llama-3.1-8B-Instruct"
ANSWER_FOLDER="./answers"
NUM=32

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --answer_folder) ANSWER_FOLDER="$2"; shift 2;;
    --num_generations) NUM="$2"; shift 2;;
    -h|--help)
      echo "Usage: bash $0 [--model <model>] [--answer_folder <folder>] [--num_generations <num>]"
      exit 0;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

echo "Running with:"
echo "  Model          : $MODEL"
echo "  Answer Folder  : $ANSWER_FOLDER"
echo "  Num Generations: $NUM"

bash scripts/answer_generate.sh "$MODEL" "answer/$ANSWER_FOLDER" $NUM 

bash scripts/confidence_calculate.bash \
  "$MODEL" \
  "answer/$ANSWER_FOLDER" \
  "confidence_result/$ANSWER_FOLDER"

bash scripts/analysis.bash "confidence_result/$ANSWER_FOLDER"


