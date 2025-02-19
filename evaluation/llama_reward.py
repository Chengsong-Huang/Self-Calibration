import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import argparse
import json
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import logging
from tqdm import tqdm
import torch.distributed as dist
from utils.dataset_loader import *
from utils.metric import *
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc
from utils.SPECIAL_SUFFIXS import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Function to add True/False prompt and calculate confidence from token probabilities
def add_new_inference_batch(new_model, prompts_with_responses, max_length=16, temperature=0.0, model_name = 'phi'):
    print('start')
    true_false_prompts = [f"{item['prompt'].replace('<|eot_id|><|start_header_id|>assistant<|end_header_id|>','')} {item['response']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" for item in prompts_with_responses]
    sampling_params = SamplingParams(max_tokens=1, temperature=temperature, logprobs=20)
    outputs = new_model.generate(true_false_prompts, sampling_params)
    confidences = []
    true_probs = []
    false_probs = []
    for output in outputs:
        logprobs = output.outputs[0].logprobs[0]
        true_prob = 0
        true_prob += sum(np.exp(logprob.logprob) for token_id, logprob in logprobs.items() 
            if '+' in logprob.decoded_token) 
        # print(true_prob)
        confidence = true_prob
        confidences.append(confidence)
    assert len(confidences) == len(prompts_with_responses)
    assert len(outputs) == len(confidences) 
    return confidences, true_probs, false_probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate confidence scores from saved responses.")
    parser.add_argument("--responses_file", type=str, required=True, help="Path to the input folder containing saved responses and dataset.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the model for new inference.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder to save results and plots.")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the output folder to save results and plots.")
    parser.add_argument("--model_type", type=str, required=True, help="Path to the output folder to save results and plots.")

    args = parser.parse_args()
    
    # Define file paths
    responses_file = args.responses_file
    dataset_file = args.dataset_name
    output_file = os.path.join(args.output_folder, "results_with_reward.json")

    # Ensure output folder exists


    if os.path.exists(args.output_folder):
        print(f"Folder '{args.output_folder}' already exists. Executing other code...")
        # exit(0)
    else:
        os.makedirs(args.output_folder)
        print(f"Folder '{args.output_folder}' created successfully.")

    
    # Load dataset handler
    dataset_handler = get_dataset(dataset_file)
    # Load saved responses
    with open(responses_file, "r") as f:
        all_responses = json.load(f)
    
    # Initialize new model for generating confidence scores
    new_model = LLM(model='RLHFlow/Llama3.1-8B-ORM-Mistral-Data', gpu_memory_utilization=0.95)
    
    # Generate confidence scores using model B
    confidences, true_probs, false_probs = add_new_inference_batch(new_model, all_responses, temperature=0.0, model_name=args.model_type)
    
    # Prepare results with confidence scores
    results = []
    for j, item in enumerate(all_responses):
        correct_answer = dataset_handler.extract_answer(item['correct_answer_text'])
        answer = dataset_handler.extract_answer(item["response"])
        is_correct = dataset_handler.check(correct_answer, answer)
        results.append({
            "index": j,
            "prompt": item["prompt"],
            "response": item["response"],
            "answer_text": item['correct_answer_text'],
            "correct_answer": correct_answer,
            "answer": answer,
            "is_correct": is_correct,
            "confidence": confidences[j],
        })
    
    auc_value, accuracy, ece = evaluate_inference(results)

    with open(output_file, "w") as f:
        json.dump({"auc": auc_value, "acc": accuracy, "ece": ece, 'results': results}, f, indent=4)
    print(f"Results with confidence scores saved to {output_file}")

    if dist.is_initialized():
        dist.destroy_process_group()

    is_correct_list = [result['is_correct'] for result in results]
    scores = [result['confidence'] for result in results]

    def plot_score_vs_accuracy(scores, is_correct_list, bin_size=0.05):
        bins = np.arange(0, 1 + bin_size, bin_size)
        bin_indices = np.digitize(scores, bins) - 1
        avg_scores = []
        avg_accuracies = []

        for i in range(len(bins) - 1):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                avg_score = np.mean([scores[j] for j in range(len(scores)) if bin_mask[j]])
                avg_accuracy = np.mean([is_correct_list[j] for j in range(len(is_correct_list)) if bin_mask[j]])
                avg_scores.append(avg_score)
                avg_accuracies.append(avg_accuracy * 100)

        hist, bin_edges = np.histogram(scores, bins=bins, density=True)
        cdf = np.cumsum(hist) * bin_size

        fig, ax1 = plt.subplots(figsize=(10, 6))


        ax1.plot(bin_edges[:-1], cdf, color='lightblue', linestyle='--', label='CDF Scores')
        ax1.set_xlabel('Average Score')
        ax1.set_ylabel('Cumulative Distribution')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.scatter(avg_scores, avg_accuracies, color='blue', label='Scores vs Accuracy')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend(loc='upper right')

        plt.title('Score vs Accuracy and CDF')
        output_path = os.path.join(args.output_folder, 'score_vs_accuracy_analysis.pdf')
        plt.savefig(output_path)
        plt.show() 

    def plot_roc_curve(scores, is_correct_list):
        y_true = np.array(is_correct_list)
        y_scores = np.array(scores)

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        output_path = os.path.join(args.output_folder, 'roc_curve_analysis.pdf')
        plt.savefig(output_path)
        plt.show()
    plot_score_vs_accuracy(scores, is_correct_list)
    plot_roc_curve(scores, is_correct_list)

    print(f"Analysis plot saved to {os.path.join(args.output_folder, 'score_vs_accuracy_analysis.pdf')}")
    print(f"ROC curve saved to {os.path.join(args.output_folder, 'roc_curve_analysis.pdf')}")
