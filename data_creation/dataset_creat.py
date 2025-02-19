import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import re
import pandas as pd
import json
import argparse
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import os
from huggingface_hub import login
from utils.dataset_loader import *


login()

# ====== 1. (Optional) Login to Hugging Face ======
# If you haven't logged in via CLI, you can do so in this script:
# login("hf_your_access_token")

# =============== Utility functions ===============

def process_data(d: str, model):
    """
    Read a JSON file assumed to have the structure:
    {
       "results": [
          {
             "prompt": "...",
             "responses": [...],
             "confidence": [...]
          },
          ...
       ]
    }
    For each item, extract info and compute:
      - weighted_consistency
      - consistency
    Return a list of records.
    """
    with open(f'result/{d}/{model}/0.8_32_train.json', 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    database = get_dataset(d)
    results = data_json['results']  # Adjust according to your JSON structure

    processed_data = []
    for item in results:
        response_data = []
        for response, conf in zip(item["responses"], item["confidence"]):

            answer = database.extract_answer(response)
            if answer is None:
                continue
            response_data.append({
                "input": f"{item['prompt']} {response}",
                "answer": answer,
                "confidence": conf
            })

        df = pd.DataFrame(response_data)

        # Compute weighted_consistency
        try:
            weighted_confidence_summary = df.groupby("answer")["confidence"].sum() / df["confidence"].sum()
        except:
            print(df)
            continue
        df["weighted_consistency"] = df["answer"].map(weighted_confidence_summary)

        # Compute unweighted consistency
        unweighted_counts = df["answer"].value_counts(normalize=True)
        df["consistency"] = df["answer"].map(unweighted_counts)

        # Remove the confidence column
        df = df.drop(columns=['confidence'])

        # Extend the final list
        processed_data.extend(df.to_dict(orient="records"))

    return processed_data

def save_to_jsonl(data, output_path):
    """
    Save a list of dictionaries to a JSONL file.
    Each line is a valid JSON object.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# =============== Main execution flow ===============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple JSON files and upload them to a single HF Dataset repo, each as a different config.")
    parser.add_argument(
        "--paths", 
        nargs='+', 
        required=True, 
        help="One or more JSON file paths, separated by space. Example: --paths file1.json file2.json ..."
    )
    parser.add_argument(
        "--repo_name", 
        required=True, 
        help="Name of the Hugging Face Dataset repository, e.g. 'your-username/my-dataset'"
    )
    parser.add_argument(
        "--model", 
        required=True, 
        help="Name of the model, e.g. 'your-username/my-dataset'"
    )
    parser.add_argument(
        "--test_ratio", 
        type=float, 
        default=0.1, 
        help="Ratio of test split; default is 0.1"
    )
    parser.add_argument(
        "--delete_local_files", 
        action="store_true",
        help="Whether to delete the temporary JSONL files after upload."
    )
    args = parser.parse_args()

    
    # Iterate through each file path, process data, split, and upload
    for idx, path in enumerate(args.paths):
        # 1) Process the data
        processed_data = process_data(path, args.model)

        # 2) Train/test split
        train_data, test_data = train_test_split(processed_data, test_size=args.test_ratio, random_state=42)

        # 3) Save to local JSONL files (temporary, just to create a HF Dataset)
        train_file = f"train_{idx}.jsonl"
        test_file = f"test_{idx}.jsonl"
        save_to_jsonl(train_data, train_file)
        save_to_jsonl(test_data, test_file)

        # 4) Create a DatasetDict from the JSON files
        train_dataset = Dataset.from_json(train_file)
        test_dataset = Dataset.from_json(test_file)
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })

        # 5) Push to the same repo but with a different config_name
        config_name = path  # You can replace this with any meaningful name
        print(f"Uploading {path} as config={config_name} to repo={args.repo_name} ...")
        
        dataset_dict.push_to_hub(
            repo_id=args.repo_name,
            # private=True,
            config_name=config_name
        )
        print(f"Finished uploading {path} -> config={config_name}\n")

        # 6) (Optional) Delete local files
        if args.delete_local_files:
            try:
                os.remove(train_file)
                os.remove(test_file)
                print(f"Temporary files {train_file} and {test_file} have been deleted.")
            except Exception as e:
                print(f"Error deleting temp files: {e}")
    
    print("All subdatasets have been uploaded successfully!")
