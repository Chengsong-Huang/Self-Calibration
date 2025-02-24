import argparse
import json
import re
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
import gc
from utils.dataset_loader import *
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import logging
from tqdm import tqdm
from utils.metric import evaluate
from data_creation.dt_generator import EntropyDynamicTemperatureGenerator  # Import the dynamic temperature generator
import torch
from utils.SPECIAL_SUFFIXS import *
# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to generate responses using dynamic temperature
def generate_response_batch(prompts, generator, max_length=1024):
    outputs = []
    for prompt in tqdm(prompts, desc="Generating Responses"):
        response = generator.generate(prompt, max_tokens=max_length)
        outputs.append(response)
    return outputs

# Function to add True/False prompt and calculate confidence from token probabilities
def add_true_false_prompt_and_calculate_confidence(llm, input_prompt, response, temperature=0.0, model_name = 'phi'):
    SPECIAL_SUFFIX = get_suffix(model_name)
    true_false_prompt = f"{input_prompt} {response} {SPECIAL_SUFFIX}"
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    output = llm.generate([true_false_prompt], sampling_params, use_tqdm=False)[0]
    logprobs = output.outputs[0].logprobs[0]
    true_probs = sum(np.exp(logprob.logprob) for token_id, logprob in logprobs.items() if 'Yes' in logprob.decoded_token)
    return true_probs

def self_consistency_check_batch(prompts, generator, dataset_handler, num_generations=5):
    all_responses = [generate_response_batch(prompts, generator) for _ in range(num_generations)]
    all_responses = np.array(all_responses).T.tolist()
    results = []


    # print(f"Allocated before: {torch.cuda.memory_allocated() / 1e6} MB")
    # print(f"Reserved before: {torch.cuda.memory_reserved() / 1e6} MB")

    generator.delete()
    gc.collect()
    torch.cuda.empty_cache()

    # print(f"Allocated after: {torch.cuda.memory_allocated() / 1e6} MB")
    # print(f"Reserved after: {torch.cuda.memory_reserved() / 1e6} MB")


    # Load the LLM for evaluating the response (frees up memory for VLLM loading)
    llm = LLM(model=args.model_name, gpu_memory_utilization=0.95, tensor_parallel_size=args.n_GPU)
    for j, responses in tqdm(enumerate(all_responses), total=len(all_responses)):
        matched_answers = []
        confidences = []

        for response in responses:
            confidence = add_true_false_prompt_and_calculate_confidence(llm, prompts[j], response, model_name=args.model_name)
            confidences.append(confidence)
            answer_content = dataset_handler.extract_answer(response)
            matched_answers.append(answer_content)
        if matched_answers:
            weighted_counts = {ans: matched_answers.count(ans) for ans in set(matched_answers)}
            weighted_counts_c = {ans: sum(c for n, c in zip(matched_answers, confidences) if n == ans) for ans in set(matched_answers)}
            most_common_response = max(weighted_counts, key=weighted_counts.get)
            most_common_response_c = max(weighted_counts_c, key=weighted_counts_c.get)
            total_weight = sum(confidences)
            consistency_score = weighted_counts[most_common_response] / len(confidences) if total_weight > 0 else 0
            consistency_score_c = weighted_counts_c[most_common_response_c] / total_weight if total_weight > 0 else 0
        else:
            most_common_response, most_common_response_c, consistency_score = None, None, 0

        results.append((most_common_response, most_common_response_c, consistency_score, consistency_score_c, responses, confidences))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with specified temperature and number of generations.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to be used.")
    parser.add_argument("--use_cot", action="store_true", help="Use chain-of-thought format.")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("--num_generations", type=int, help="Number of generations per prompt")
    parser.add_argument("--subset", type=str, default='train', help="Subset of the data to use (e.g., 'train', 'test')")
    parser.add_argument("--data_size", type=int, default=None, help="Number of data points to use.")
    parser.add_argument("--n_GPU", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--suffix", type=int, default=0, help="Number of GPUs to use.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for response generation.")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save results with confidence scores.")




    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    generator = EntropyDynamicTemperatureGenerator(model_name=args.model_name)

    dataset_handler = get_dataset(args.dataset_name)

    data, answer_type = dataset_handler.load_data()
    data = data[args.subset]
    if args.data_size:
        data = data.select(range(args.data_size*args.suffix,args.data_size*(args.suffix+1)))
    qa_data = dataset_handler.prepare_qa_data(data)

    prompts, qa_data = generate_prompt(args, logger, qa_data, answer_type=answer_type, tokenizer=tokenizer)

    batch_results = self_consistency_check_batch(prompts, generator, dataset_handler, num_generations=args.num_generations)
    results = []

    for j, (most_common_response, most_common_response_c, consistency_score, consistency_score_c, responses, confidences) in enumerate(batch_results):
        correct_answer_text = list(qa_data.values())[j]
        correct_answer = dataset_handler.extract_answer(correct_answer_text)
        is_correct = (most_common_response == correct_answer) if correct_answer else False
        is_correct_c = (most_common_response_c == correct_answer) if correct_answer else False
        results.append({
            "index": j,
            "prompt": prompts[j],
            "most_common_response": most_common_response,
            "consistency_score": consistency_score,
            "most_common_response_c": most_common_response_c,
            "consistency_score_c": consistency_score_c,
            "correct_answer": correct_answer if correct_answer else "N/A",
            "is_correct": is_correct,
            "is_correct_c": is_correct_c,
            "responses": responses,
            "confidence": confidences
        })

    auc, auc_c, accuracy, accuracy_c, ece, ece_c = evaluate(results)
    filename = f"result/{args.dataset_name}/{args.save_path}/{args.temperature}_{args.num_generations}_{args.subset}.json"
    folder_path = os.path.dirname(filename)
    os.makedirs(folder_path, exist_ok=True)
    with open(filename, "a") as f:
        json.dump({
            "num_generations": args.num_generations,
            "auc": auc,
            "auc_c": auc_c,
            "accuracy": accuracy,
            "accuracy_c": accuracy_c,
            "ece": ece,
            "ece_c": ece_c,
            "results": results
        }, f, indent=4)
        f.write('\n')
    print(f"Results saved to {filename}")
