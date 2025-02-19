#!/bin/bash


if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <model> <answer_folder> <number_generation>"
  exit 1
fi

MODEL=$1
ANSWER_FOLDER=$2
NUM=$3

bash scripts/answer_generate.sh "$MODEL" "answer/$ANSWER_FOLDER" $NUM 

bash scripts/confidence_calculate.bash \
  "$MODEL" \
  "answer/$ANSWER_FOLDER" \
  "confidence_result/$ANSWER_FOLDER"

bash scripts/analysis.bash "confidence_result/$ANSWER_FOLDER"


