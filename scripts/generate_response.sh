#!/bin/bash

# Read command-line arguments for model name, dataset name, final output file, and other parameters
while getopts m:d:o:g:t:s: flag
do
    case "${flag}" in
        m) MODEL_NAME=${OPTARG};;
        d) DATASET_NAME=${OPTARG};;
        o) FINAL_OUTPUT=${OPTARG};;
        g) NUM_GENERATIONS=${OPTARG};;
        t) TEMPERATURE=${OPTARG};;
        s) NUM_SUBSETS=${OPTARG};;
    esac
done

# Ensure the output directory for FINAL_OUTPUT exists
OUTPUT_DIR=$(dirname "$FINAL_OUTPUT")
mkdir -p "$OUTPUT_DIR"

# Run the script on multiple GPUs, each processing a different subset
for i in $(seq 0 $((NUM_SUBSETS - 1)))
do
  OUTPUT_FILE="output_subset_$i.json"
  CUDA_VISIBLE_DEVICES=$i python evaluation/generate_responses.py \
    --model_name $MODEL_NAME \
    --temperature $TEMPERATURE \
    --dataset_name $DATASET_NAME \
    --num_generations $NUM_GENERATIONS \
    --output_file $OUTPUT_FILE \
    --subset_index $i \
    --num_subsets $NUM_SUBSETS \
    --use_cot &
done

wait

echo "All subsets processed. Now merging the results."

# Merge all subsets into a single output file
python - <<EOF
import json
import os

output_files = ["output_subset_{}.json".format(i) for i in range($NUM_SUBSETS)]
all_results = []

for file in output_files:
    with open(file, "r") as f:
        all_results.extend(json.load(f))

output_dir = os.path.dirname("$FINAL_OUTPUT")
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

with open("$FINAL_OUTPUT", "w") as f:
    json.dump(all_results, f, indent=4)

print("All subsets merged into $FINAL_OUTPUT")
EOF

# Remove the temporary subset files
for i in $(seq 0 $((NUM_SUBSETS - 1)))
do
  rm "output_subset_$i.json"
done

echo "Temporary files removed. Process completed."
