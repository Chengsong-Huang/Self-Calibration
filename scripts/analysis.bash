#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <folder_name>"
    exit 1
fi

folder_name=$1

if [ ! -d "$folder_name" ]; then
    echo "Error: Folder '$folder_name' does not exist."
    exit 1
fi

for subfolder in "$folder_name"/*; do
    if [ -d "$subfolder" ]; then
        subfolder_name=$(basename "$subfolder")
        echo "Processing subfolder: $subfolder_name"
        python evaluation/analysis.py --input_file "$folder_name/$subfolder_name" --dataset_name $subfolder_name
    fi
done
