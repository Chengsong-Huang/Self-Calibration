import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))


import argparse
import json
import re
import numpy as np
from utils.dataset_loader import *
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import logging
from tqdm import tqdm
from utils.metric import evaluate
from sklearn.metrics import roc_auc_score, accuracy_score

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to generate responses
def generate_response_batch(prompts, llm, num_generations=5, max_length=1024, temperature=0.8):
    sampling_params = SamplingParams(max_tokens=max_length, temperature=temperature, n=num_generations)
    outputs = llm.generate(prompts, sampling_params)
    return [[output.text for output in result.outputs] for result in outputs]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with specified temperature and number of generations.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to be used.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for response generation.")
    parser.add_argument("--use_cot", action="store_true", help="Use chain-of-thought format.")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("--num_generations", type=int, default=5, help="Number of answers to generate per question.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save generated responses.")
    parser.add_argument("--subset_index", type=int, default=0, help="Index of the subset to process (used for splitting the data across multiple runs).")
    parser.add_argument("--num_subsets", type=int, default=1, help="Total number of subsets to split the data into.")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Choose dataset handler (example: GSM8kHandler, extend with other handlers)
    dataset_handler = get_dataset(args.dataset_name)

    # Load dataset and prepare QA data
    data, answer_type = dataset_handler.load_data()
    data = data['test']
    if len(data) > 5000:
        data = data.select(range(5000))
    qa_data = dataset_handler.prepare_qa_data(data)
    
    # Split data into subsets
    total_questions = len(qa_data)
    subset_size = total_questions // args.num_subsets
    start_index = args.subset_index * subset_size
    end_index = total_questions if args.subset_index == args.num_subsets - 1 else (args.subset_index + 1) * subset_size
    qa_data_subset = {k: qa_data[k] for k in list(qa_data.keys())[start_index:end_index]}
    
    # Generate prompts with tokenizer
    prompts, qa_data_subset = generate_prompt(args, logger, qa_data_subset, answer_type=answer_type, tokenizer=tokenizer)
    # Initialize model and tokenizer
    llm = LLM(model=args.model_name, gpu_memory_utilization=0.95)
    
    # Generate responses using model
    all_responses = generate_response_batch(prompts, llm, num_generations=args.num_generations, temperature=args.temperature)
    results = []
    for j in range(len(all_responses)):
        for i, response in enumerate(all_responses[j]):
            results.append({
                "prompt": prompts[j],
                "response": response,
                "correct_answer_text": list(qa_data_subset.values())[j],
                "generation_index": i
            })
    
    responses_filename = args.output_file
    with open(responses_filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Responses saved to {responses_filename}")
