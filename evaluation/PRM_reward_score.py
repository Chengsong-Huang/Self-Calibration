import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
import argparse
from utils.dataset_loader import *
from utils.metric import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from utils.SPECIAL_SUFFIXS import *
model_name = "Qwen/Qwen2.5-Math-PRM-7B"
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model = model.to("cuda:0")
model = nn.DataParallel(model)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def make_step_rewards_batch(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]
        nonzero_probs = sample[sample != 0].view(-1, 2)[:, 1]
        non_zero_elements_list = nonzero_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

def build_input_strings(datas):
    input_strings = []
    for data in datas:
        messages = [
            {"role": "system", "content": data['system']},
            {"role": "user", "content": data['query']},
            {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
        ]
        conversation_str = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        input_strings.append(conversation_str)
    return input_strings

def get_step_rewards_batch(datas, batch_size=32):
    step_sep_id = tokenizer.encode("<extra_0>", add_special_tokens=False)[0]
    final_rewards = []
    whole = []
    for i in tqdm(range(0, len(datas), batch_size)):
        batch_data = datas[i : i + batch_size]
        input_strs = build_input_strings(batch_data)
        encodings = tokenizer(
            input_strs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        for k, v in encodings.items():
            encodings[k] = v.to("cuda:0")
        with torch.no_grad():
            outputs = model(**encodings)
        token_masks = (encodings["input_ids"] == step_sep_id).long()
        step_rewards_list = make_step_rewards_batch(outputs[0], token_masks)
        for step_probs in step_rewards_list:
            min_val = min(step_probs) if len(step_probs) > 0 else None
            final_rewards.append(min_val)
            whole.append(step_probs)
    return final_rewards, whole

def get_input(result):
    prompt = result['prompt']
    response = result['response']
    system = prompt[0].replace('<|im_start|>system\n', '')
    query = prompt[1].replace('<|im_start|>user\nQuestion: Problem: ', '').replace('\n<|im_end|>\n<|im_start|>assistant\n', '')
    responses = response.split('\n\n')
    return {
        "system": system,
        "query": query,
        "response": responses
    }

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
    if os.path.exists(output_file):
        print(f"Results with confidence scores already saved to {output_file}")
        exit()
    with open(responses_file, 'r') as f:
        all_responses = json.load(f)

    datas = [get_input(r) for r in all_responses]
    confidences, whole = get_step_rewards_batch(datas, batch_size=8)
    dataset_handler = get_dataset(dataset_file)
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
            "whole": whole[j]

        })
    
    # Save results with confidence scores
    auc_value, accuracy, ece = evaluate_inference(results)

    with open(output_file, "w") as f:
        json.dump({"auc": auc_value, "acc": accuracy, "ece": ece, 'results': results}, f, indent=4)
    print(f"Results with confidence scores saved to {output_file}")