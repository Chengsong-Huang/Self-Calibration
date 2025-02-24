import re
from abc import ABC, abstractmethod
from datasets import load_dataset
import random
random.seed(77)
# ---------------------------------------------------------------------------
# 1. Abstract base class: DatasetHandler
# ---------------------------------------------------------------------------
class DatasetHandler(ABC):
    @abstractmethod
    def load_data(self):
        """
        Load the dataset and return a tuple: (splits_dict, answer_type).

        splits_dict: A dictionary where each key is a split name (e.g., 'train', 'test')
                     and the value is the corresponding dataset or data structure.
        answer_type: A string describing the type of the answer, e.g.:
                     'number', 'text', 'option letter', etc.
        """
        pass
    
    @abstractmethod
    def prepare_qa_data(self, data):
        """
        Given a particular split (like a list or IterableDataset),
        transform it into a dictionary: {prompt_text -> ground_truth_answer}.
        """
        pass

    @abstractmethod
    def extract_answer(self, response):
        """
        Given a model-generated response (string), extract the final answer
        so that it matches the ground truth format (number, letter, text, etc.).
        """
        pass

    def check(self, correct_answer, response):
        """
        Given the correct answer and the model-generated response,
        check if the response is correct. This is a simple equality check.
        """
        return correct_answer == response

# ---------------------------------------------------------------------------
# 2. Existing dataset handlers (with trust_remote_code=True where needed)
# ---------------------------------------------------------------------------

# GSM8k: math word problems (English, elementary-level arithmetic)
class GSM8kHandler(DatasetHandler):
    def load_data(self):
        return {
            'train': load_dataset("gsm8k", 'main', split="train"),
            'test': load_dataset("gsm8k", 'main', split="test")
        }, "number"
    
    def prepare_qa_data(self, data):
        # Each entry has a "question" and an "answer"
        # We'll create {question -> answer}
        return {entry['question']: entry['answer'] for entry in data}

    def extract_answer(self, response):
        # Extract the last numeric substring from the response
        numbers = re.findall(r'[-+]?\$?\d+(?:,\d{3})*(?:\.\d+)?', response)
        if numbers:
            answer = numbers[-1].replace(',', '').replace('$', '')
            if answer.endswith('.00'):
                answer = answer.replace('.00', '')
            return answer
        return None


# SciQ: science reading comprehension with multiple choice answers
class SciQHandler(DatasetHandler):
    def load_data(self):
        return {
            'train': load_dataset("allenai/sciq", split="train"),
            'validation': load_dataset("allenai/sciq", split="validation"),
            'test': load_dataset("allenai/sciq", split="test")
        }, "option letter"
    
    def prepare_qa_data(self, data):
        """
        SciQ has the following structure:
        {
          'question': ...,
          'distractor1': ...,
          'distractor2': ...,
          'distractor3': ...,
          'correct_answer': ...
        }
        We'll build a question with sorted options, then label the correct one (A/B/C/D).
        """
        qa_pairs = {}
        for entry in data:
            question = entry['question']
            # Combine distractors and correct_answer
            options = [entry['distractor1'], 
                       entry['distractor2'], 
                       entry['distractor3'], 
                       entry['correct_answer']]
            # Sort or shuffle; let's sort by text here
            enumerated_opts = list(enumerate(options))
            enumerated_opts.sort(key=lambda x: x[1])  # sort by text
            # Build prompt
            options_text = "\n".join([
                f"{chr(65+i)}. {opt}" 
                for i, (idx, opt) in enumerate(enumerated_opts)
            ])
            prompt = (
                f"Question: {question}\n"
                f"Options:\n{options_text}\n"
            )
            # Find the correct answer index in enumerated_opts
            correct_index = next(
                i for i, (idx, opt) in enumerate(enumerated_opts)
                if opt == entry['correct_answer']
            )
            correct_option = chr(65 + correct_index)
            qa_pairs[prompt] = f"Answer:{correct_option}"
        return qa_pairs

    def extract_answer(self, response):
        match = re.search(r"Answer:\s*?\(?([A-E])\)?", response)
        if match:
            return match.group(1)
        return None


# CommonsenseQA: a typical commonsense multiple-choice dataset
class CommonsenseQAHandler(DatasetHandler):
    def load_data(self):
        return {
            'train': load_dataset("tau/commonsense_qa", split="train"),
            'test': load_dataset("tau/commonsense_qa", split="validation")
        }, "option letter"
    
    def prepare_qa_data(self, data):
        """
        Each entry: question, choices['text'], answerKey (like 'A','B','C','D','E')
        We'll build a multiple-choice prompt and label the correct one with "Answer:X".
        """
        qa_pairs = {}
        for entry in data:
            question = entry['question']
            options = entry['choices']['text']  # list of possible answers
            options_text = "\n".join([
                f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)
            ])
            prompt = (
                f"Question: {question}\n"
                f"Options:\n{options_text}\n"
            )
            correct_answer = entry['answerKey']  # e.g. 'A'
            qa_pairs[prompt] = 'Answer:' + correct_answer
        return qa_pairs

    def extract_answer(self, response):
        match = re.search(r"Answer:\s*?\(?([A-E])\)?", response)
        if match:
            return match.group(1)
        return None


# ObjectCountingHandler: a specialized sub-task from BBH
class ObjectCountingHandler(DatasetHandler):
    def load_data(self):
        # This is from the BBH dataset, sub-task "object_counting"
        return {
            'test': load_dataset("lukaemon/bbh", "object_counting", split="test")
        }, "number"
    
    def prepare_qa_data(self, data):
        """
        Each entry has an 'input' (the description) and 'target' (the numeric answer).
        We'll prompt: "Please count the objects... The answer should be a number."
        """
        qa_pairs = {}
        for entry in data:
            prompt = (
                f"Please count the number of objects in the following description:\n"
                f"{entry['input']}\n"
            )
            qa_pairs[prompt] = entry['target']
        return qa_pairs

    def extract_answer(self, response):
        # Attempt to parse the last integer from the response
        numbers = re.findall(r'\d+', response)
        if numbers:
            return int(numbers[-1])
        return None

class GPQAHandler(DatasetHandler):
    def load_data(self):
        dataset = load_dataset("Idavidrein/gpqa", 'gpqa_diamond')
        return {
            'test': dataset['train']
        }, "option letter"
    
    def prepare_qa_data(self, data):
        qa_pairs = {}
        for entry in data:
            question = entry['Question']
            options = [
                entry['Correct Answer'],
                entry['Incorrect Answer 1'],
                entry['Incorrect Answer 2'],
                entry['Incorrect Answer 3']
            ]
            random.shuffle(options)
            options_text = "\n".join([
                f"{chr(65+i)}. {opt}"
                for i, opt in enumerate(options)
            ])
            prompt = (
                f"{question}\n"
                f"Options:\n{options_text}\n"
            )
            correct_index = options.index(entry['Correct Answer'])
            correct_option = chr(65 + correct_index)
            qa_pairs[prompt] = f"Answer: {correct_option}"
        return qa_pairs

    def extract_answer(self, response):
        match = re.search(r"Answer:\s*?\(?([A-E])\)?", response)
        if match:
            return match.group(1)
        return None

# WinoGrande: a pronoun coreference challenge dataset
class WinoGrandeHandler(DatasetHandler):
    def load_data(self):
        dataset = load_dataset("allenai/winogrande", "winogrande_xl")
        return {
            'train': dataset['train'],
            'test': dataset['validation']
        }, "option letter"
    
    def prepare_qa_data(self, data):
        """
        Each entry: sentence, option1, option2, answer ('1' or '2').
        We'll convert '1'->'A' and '2'->'B'.
        """
        qa_pairs = {}
        for entry in data:
            sentence = entry['sentence']
            opt1 = entry['option1']
            opt2 = entry['option2']
            correct = entry['answer']
            if correct == "1":
                correct_label = "A"
            elif correct == "2":
                correct_label = "B"
            else:
                continue
            prompt = (
                f"Question: {sentence}\n"
                "Options:\n"
                f"A. {opt1}\n"
                f"B. {opt2}\n"
            )
            qa_pairs[prompt] = f"Answer:{correct_label}"
        return qa_pairs

    def extract_answer(self, response):
        match = re.search(r"Answer:\s*?\(?([A-D])\)?", response)
        if match:
            return match.group(1)
        return None


# OpenBookQA: science QA with a small "open book" of scientific facts
class OpenBookQAHandler(DatasetHandler):
    def load_data(self):
        dataset = load_dataset("allenai/openbookqa", "main")
        return {
            'train': dataset['train'],
            'validation': dataset['validation'],
            'test': dataset['test']
        }, "option letter"
    
    def prepare_qa_data(self, data):
        """
        Each entry: question_stem, choices (text/label), answerKey (like 'A','B','C','D')
        We'll do a standard multiple-choice prompt.
        """
        qa_pairs = {}
        for entry in data:
            question = entry['question_stem']
            choices = entry['choices']['text']
            labels = entry['choices']['label']
            correct_answer = entry['answerKey']  # e.g. 'C'
            
            options_text = "\n".join([
                f"{label}. {choice}" 
                for label, choice in zip(labels, choices)
            ])
            prompt = (
                f"Question: {question}\n"
                f"Options:\n{options_text}\n"
            )
            qa_pairs[prompt] = "Answer:" + correct_answer
        return qa_pairs

    def extract_answer(self, response):
        match = re.search(r"Answer:\s*?\(?([A-D])\)?", response)
        if match:
            return match.group(1)
        return None


# ReClor: reading comprehension with a focus on logical reasoning
class ReClorHandler(DatasetHandler):
    def load_data(self):
        # NOTE: Pass trust_remote_code=True to handle custom code for ReClor
        dataset = load_dataset("metaeval/reclor", trust_remote_code=True)
        return {
            'train': dataset['train'],
            'test': dataset['validation']
        }, "option letter"
    
    def prepare_qa_data(self, data):
        """
        Each entry: context (passage), question, answers (list), label (0..3).
        We'll map label 0..3 to A..D.
        """
        qa_pairs = {}
        for entry in data:
            passage = entry['context']
            question = entry['question']
            answers = entry['answers']
            label_index = entry['label']
            if 0 <= label_index < len(answers):
                correct_label = chr(65 + label_index)
            else:
                continue
            options_text = "\n".join([
                f"{chr(65+i)}. {ans}" for i, ans in enumerate(answers)
            ])
            prompt = (
                f"Passage:\n{passage}\n\n"
                f"Question: {question}\n\n"
                f"Options:\n{options_text}\n"
            )
            qa_pairs[prompt] = f"Answer:{correct_label}"
        return qa_pairs

    def extract_answer(self, response):
        match = re.search(r"Answer:\s*?\(?([A-D])\)?", response)
        if match:
            return match.group(1)
        return None

# MathQA: math word problems from "math_qa"
class MathQAHandler(DatasetHandler):
    def load_data(self):
        return {
            'train': load_dataset("allenai/math_qa", split="train",trust_remote_code=True),
            'validation': load_dataset("allenai/math_qa", split="validation",trust_remote_code=True),
            'test': load_dataset("allenai/math_qa", split="test",trust_remote_code=True)
        }, "number"
    
    def prepare_qa_data(self, data):
        """
        data: A list (or Dataset) of entries, each like:
          {
            'Problem': "...",
            'Rationale': "...",
            'options': "a ) ... , b ) ... , c ) ... , d ) ... , e ) ...",
            'correct': "a",
            ...
          }
        We'll convert each entry to a multiple-choice style prompt:
          prompt = "Problem: ...\nOptions:\nA) ...\nB) ...\nC) ...\n..."
          answer = "Answer:A"
        """
        qa_pairs = {}

        for entry in data:
            problem_text = entry.get('Problem', "")
            options_str = entry.get('options', "")
            correct_label = entry.get('correct', "").lower().strip()  
            raw_opts = [opt.strip() for opt in options_str.split(',')]
            formatted_opts = []
            for opt in raw_opts:
                match = re.match(r"([a-z])\s*\)", opt, flags=re.IGNORECASE)
                if match:
                    letter = match.group(1).upper()
                    text_part = opt[match.end():].strip()
                    formatted_opts.append(f"{letter}. {text_part}")
                else:
                    # fallback
                    formatted_opts.append(opt)

            # Now build the final Options text block
            options_block = "\n".join(formatted_opts)

            # Build the prompt
            prompt = (
                f"Problem: {problem_text}\n"
                f"Options:\n{options_block}\n"
            )

            # Convert the correct letter to uppercase
            correct_letter = correct_label.upper()
            # The ground truth answer for QA
            gt_answer = f"Answer:{correct_letter}"

            qa_pairs[prompt] = gt_answer
        
        return qa_pairs

    def extract_answer(self, response):
        """
        We'll look for something like "Answer:A" or "Answer: B" in the response.
        If found, return that letter. Otherwise return None.
        """
        match = re.search(r"Answer:\s*([A-E])", response, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None


# ARC (AI2 Reasoning Challenge): multiple-choice science questions
class ARCHandler(DatasetHandler):
    def __init__(self, subset='ARC-Challenge'):
        self.subset = subset

    def load_data(self):
        dataset = load_dataset("ai2_arc", self.subset)
        return {
            'train': dataset['train'],
            'validation': dataset['validation'],
            'test': dataset['test']
        }, "option letter"
    
    def prepare_qa_data(self, data):
        """
        Each entry has:
          'question': ...,
          'choices': {'text': [...], 'label': [...]},
          'answerKey': 'A'/'B'/'C'/'D'
        We'll build a multiple-choice prompt.
        """
        qa_pairs = {}
        for entry in data:
            question = entry['question']
            choices_text = entry['choices']['text']
            choices_labels = entry['choices']['label']
            answer_key = entry['answerKey']
            
            options_str = "\n".join([
                f"{lbl}. {txt}"
                for lbl, txt in zip(choices_labels, choices_text)
            ])
            prompt = (
                f"Question: {question}\n"
                f"Options:\n{options_str}\n"
            )
            qa_pairs[prompt] = f"Answer:{answer_key}"
        return qa_pairs

    def extract_answer(self, response):
        match = re.search(r"Answer:\s*?\(?([A-D])\)?", response)
        if match:
            return match.group(1)
        return None


# LogiQA: a Chinese logical reasoning multiple-choice dataset
class LogiQAHandler(DatasetHandler):
    def load_data(self):
        """
        Automatically rewritten version of the LogiQAHandler load_data method.

        This method loads the 'lucasmccabe/logiqa' dataset from Hugging Face,
        returning a dictionary of splits (train, validation, test) 
        and the answer type 'option letter'.
        """
        loaded_dataset = load_dataset("lucasmccabe/logiqa")
        splits = {
            'train': loaded_dataset['train'],
            'validation': loaded_dataset['validation'],
            'test': loaded_dataset['test']
        }
        return splits, "option letter"

    def prepare_qa_data(self, data):
        """
        Automatically rewritten version of prepare_qa_data.

        Given a dataset split (e.g., train/test/validation),
        this method constructs a dictionary mapping prompts to ground-truth answers.
        
        The typical LogiQA entry includes:
          - 'context': a text passage
          - 'question': the question being asked
          - 'answers': a list of 4 possible answers (strings)
          - 'label': an integer from 0..3, indicating which answer is correct

        We convert each entry to a multiple-choice prompt and 
        store the correct answer in the format 'Answer:X' (e.g., 'Answer:A').
        """
        qa_pairs = {}
        for entry in data:
            context = entry['context']
            question = entry['query']
            answers = entry['options']
            label = entry['correct_option']

            # Ensure label is in the valid range 0..3
            if 0 <= label < len(answers):
                correct_letter = chr(65 + label)  # 0 -> A, 1 -> B, etc.
            else:
                # If label is invalid, skip this entry
                continue

            # Build the multiple-choice options block
            options_text = "\n".join(
                f"{chr(65 + i)}. {ans}" for i, ans in enumerate(answers)
            )

            # Construct the final prompt
            prompt = (
                f"Article:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Options:\n{options_text}\n"
            )

            # Prepare the ground truth answer
            ground_truth = f"Answer:{correct_letter}"
            qa_pairs[prompt] = ground_truth

        return qa_pairs

    def extract_answer(self, response):
        match = re.search(r"Answer:\s*([A-D])", response)
        if match:
            return match.group(1).upper()
        return None


class SVAMPHandler(DatasetHandler):
    def load_data(self):
        """
        Loads the SVAMP dataset from Hugging Face.
        """
        dataset = load_dataset("ChilleD/SVAMP")
        return {
            'train': dataset['train'],
            'test': dataset['test']
        }, "number"

    def prepare_qa_data(self, data):
        """
        Prepares SVAMP QA data.
        Each entry includes:
          - 'Question': the math word problem
          - 'Answer': the numeric answer
        """
        qa_pairs = {}
        for entry in data:
            question = entry["Body"] + entry["Question"] # Math word problem
            answer = str(entry["Answer"])  # Convert answer to string for consistency
            prompt = (
                f"Question: {question}\n"
            )
            qa_pairs[prompt] = f"Answer:{answer}"
        return qa_pairs

    def extract_answer(self, response):
        """
        Extracts a numeric answer from the model's response.
        """
        match = re.search(r"Answer:\s*([-+]?\d+(?:\.\d+)?)", response)
        if match:
            return match.group(1)
        return None

class AQUARATHandler(DatasetHandler):
    def load_data(self):
        """
        Loads the AQUA-RAT dataset from Hugging Face.
        """
        dataset = load_dataset("aqua_rat")
        return {
            'train': dataset['train'],
            'validation': dataset['validation'],
            'test': dataset['test']
        }, "option letter"

    def prepare_qa_data(self, data):
        """
        Prepares AQUA-RAT QA data.
        Each entry includes:
          - 'question': the math word problem
          - 'options': the list of multiple-choice answers
          - 'correct': the correct option (e.g., 'A', 'B', etc.)
        """
        qa_pairs = {}
        for entry in data:
            question = entry["question"]
            options = entry["options"]  # AQUA-RAT options are pipe-separated
            correct_option = entry["correct"]  # Correct answer as a letter

            # Format the options
            options_text = "\n".join(
                [f"({opt}" for i, opt in enumerate(options)]
            )

            # Create the prompt
            prompt = (
                f"Question: {question}\n"
                f"Options:\n{options_text}\n"
            )
            qa_pairs[prompt] = f"Answer:{correct_option}"
        return qa_pairs

    def extract_answer(self, response):
        """
        Extracts the selected option letter from the model's response.
        """
        match = re.search(r"Answer:\s*?\(?([A-E])\)?", response)
        if match:
            return match.group(1).upper()
        return None


# ---------------------------------------------------------------------------
# 4. get_dataset function
# ---------------------------------------------------------------------------
def get_dataset(dataset_name):
    """
    Returns a DatasetHandler instance corresponding to dataset_name.
    """
    if dataset_name == 'gsm8k':
        return GSM8kHandler()
    elif dataset_name == 'sciq':
        return SciQHandler()
    elif dataset_name == 'commonsense_qa':
        return CommonsenseQAHandler()
    elif dataset_name == 'object_counting':
        return ObjectCountingHandler()
    elif dataset_name == 'winogrande':
        return WinoGrandeHandler()
    elif dataset_name == 'openbookqa':
        return OpenBookQAHandler()
    elif dataset_name == 'reclor':
        return ReClorHandler()
    elif dataset_name == 'math_qa':
        return MathQAHandler()
    elif dataset_name == 'arc_challenge':
        return ARCHandler(subset='ARC-Challenge')
    elif dataset_name == 'arc_easy':
        return ARCHandler(subset='ARC-Easy')
    elif dataset_name == 'logiqa':
        return LogiQAHandler()
    elif dataset_name == 'aqua_rat':
        return AQUARATHandler()
    elif dataset_name == 'svamp':
        return SVAMPHandler()
    elif dataset_name == 'gpqa':
        return GPQAHandler()
    else:
        raise ValueError("Unrecognized dataset name: " + dataset_name)


# ---------------------------------------------------------------------------
# 5. Example main/test code
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Demonstration: we will load each dataset, prepare a small subset of QA data,
    and print a couple of examples.

    Note: Some datasets are quite large. You may want to comment out certain lines
    if you only want to test one or two. Also, downloading everything can take a while.
    """

    dataset_names = [
        # "gsm8k",
        # "math",
        # "sciq",
        # "commonsense_qa",
        # "winogrande",
        # "openbookqa",
        # "hotpotqa",
        # "reclor",
        # "math_qa",
        # "arc_challenge",
        # "arc_easy",
        # "logiqa",
        # "svamp",
        "aqua_rat",
        # "bbh",
        "gpqa",
        "aime"
        # "mmlu",
        # "bbh_mmlu"  # our merged example
    ]

    for ds_name in dataset_names:
        print(f"\n=== Testing dataset: {ds_name} ===")
        handler = get_dataset(ds_name)

        data_splits, answer_type = handler.load_data()
        print(f"Loaded splits: {list(data_splits.keys())}, answer type: {answer_type}")

        # We'll take the first split
        first_split = list(data_splits.keys())[0]
        ds_split = data_splits[first_split]

        # For the merged BBH+MMLU, ds_split is a dict {'bbh':..., 'mmlu':...}.
        # For regular datasets, ds_split is a Dataset object.
        qa_data = handler.prepare_qa_data(ds_split)
        print(f"Prepared {len(qa_data)} QA pairs in split '{first_split}'")

        # Print a couple of sample prompts
        i = 0
        for prompt, correct_answer in qa_data.items():
            print(f"\nPrompt {i+1}:\n{prompt}\nGround Truth Answer: {correct_answer}")
            extracted = handler.extract_answer(correct_answer)
            print(f"Extracted Answer => {extracted}")
            i += 1
            if i >= 2:
                break
        
        print(f"Done testing '{ds_name}' on split '{first_split}'.")
        # Uncomment the break if you only want to test the first dataset
        # break


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