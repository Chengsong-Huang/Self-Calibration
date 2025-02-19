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

# Function to generate prompts using chat template
def generate_prompt(args, logger, qa_data, answer_type="number", tokenizer=None):
    if answer_type == "option letter":
        demo = '(A)'
    elif answer_type == "number":
        demo = 1
    elif answer_type == "latex_compression":
        demo = '\\frac{3}{2}'
    else:
        demo = '(A)'

    if args.use_cot:
        logger.info("Using COT format for answers")
        PROMPT = (
            "For the following question, provide a step-by-step explanation of your thought process.\n"
            "Use the format demonstrated below for your response.\n"
            "```Example Format:\n"
            "Explanation: <Your detailed explanation here, outlining how you arrived at your answer.>\n"
            f"Answer: <Insert your concise answer here, which should include a {answer_type} (e.g., {demo})>\n"
            "Ensure that your response strictly adheres to this format. Explicitly include the words 'Explanation:', 'Answer:'."
        ).strip()
    else:
        logger.info("Using standard format for answers")
        PROMPT = (
            f"For the following question, provide your answer including only the {answer_type}.\n"
            "Do not include any additional text, characters, or explanations. Use the format demonstrated below for your response.\n"
            "```Example Format:\n"
            f"Answer: <Insert only the {answer_type} here (e.g., {demo})>\n"
            f"Ensure that your response strictly adheres to this format and contain only the {answer_type}. Explicitly include the words 'Answer:'in your response."
        ).strip()

    prompts = []
    assert len(qa_data) > 0, "No data found"

    for question in qa_data.keys():
        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            # Construct the chat-based prompt using chat template
            chat = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": f"Question: {question}"},
            ]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True , add_special_tokens=True)
        else:
            logger.info("no chat template")
            prompt = f"{PROMPT}\n\nQuestion: {question}"

        prompts.append(prompt)
    assert len(prompts) == len(qa_data), f"Prompt generation failed. Expected {len(qa_data)} prompts, got {len(prompts)}"
    logger.info(f"Sample prompt: {prompts[0]}")
    return prompts, qa_data

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
