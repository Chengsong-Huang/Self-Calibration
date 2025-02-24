import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
import argparse
from utils.SPECIAL_SUFFIXS import *
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from utils.dataset_loader import *
import logging
from tqdm import tqdm
from tqdm.contrib import tzip
# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ===== 1) Aggregators (同之前, 略有修改以支持 conf / no conf) =====

class AggregatorBase:
    """Base class for aggregator states."""
    def __init__(self, name):
        self.name = name
        self.done = False
        self.final_answer = None
        self.used = 0

    def add_sample(self, answer: str, conf: float):
        """
        Process one new sample (answer, conf).
        Return True if aggregator decides to stop now.
        'answer' should be the *parsed* answer or raw text 
        'conf' is the numeric confidence (0.0 if aggregator doesn't use it)
        """
        raise NotImplementedError

    def finalize(self):
        """If still not done after all samples, do fallback."""
        pass


class EarlyStopAggregator(AggregatorBase):
    """
    Early Stopping (threshold-based).
    If conf >= threshold, we stop immediately.
    Otherwise fallback to the *last* sample if never triggered threshold.
    """
    def __init__(self, threshold=0.8):
        super().__init__("earlyexit")
        self.threshold = threshold
        self.all_samples = []

    def add_sample(self, answer: str, conf: float):
        if self.done:
            return True
        self.used += 1
        self.all_samples.append((answer, conf))
        if conf >= self.threshold:
            self.done = True
            self.final_answer = answer
            return True
        return False

    def finalize(self):
        if not self.done and self.all_samples:
            self.done = True
            self.final_answer = self.all_samples[-1][0]


class AscConfAggregator(AggregatorBase):
    """
    ASC w/ conf. (Weighted Dynamic Voting).
    Keep track of conf sums for each answer. If top's ratio >= threshold => stop.
    """
    def __init__(self, threshold=0.8):
        super().__init__("asc_conf")
        self.threshold = threshold
        self.conf_map = {}
        self.total_conf = 0.0

    def add_sample(self, answer: str, conf: float):
        if self.done:
            return True
        self.used += 1
        self.conf_map[answer] = self.conf_map.get(answer, 0.0) + conf
        self.total_conf = sum(self.conf_map.values())

        if self.used > 1 and self.total_conf > 0:
            top_ans = max(self.conf_map, key=self.conf_map.get)
            top_val = self.conf_map[top_ans]
            ratio = top_val / self.total_conf
            if ratio >= self.threshold:
                self.done = True
                self.final_answer = top_ans
                return True
        return False

    def finalize(self):
        if not self.done and self.conf_map:
            best_ans = max(self.conf_map, key=self.conf_map.get)
            self.done = True
            self.final_answer = best_ans


class AscAggregator(AggregatorBase):
    """
    ASC (unweighted).
    Tally freq of each answer, if top freq / total >= threshold => stop.
    """
    def __init__(self, threshold=0.8):
        super().__init__("asc")
        self.threshold = threshold
        self.count_map = {}
        self.total_count = 0

    def add_sample(self, answer: str, conf: float):
        if self.done:
            return True
        self.used += 1
        self.count_map[answer] = self.count_map.get(answer, 0) + 1
        self.total_count += 1

        if self.used>1:
            top_ans = max(self.count_map, key=self.count_map.get)
            top_cnt = self.count_map[top_ans]
            ratio = top_cnt/self.total_count
            if ratio >= self.threshold:
                self.done = True
                self.final_answer = top_ans
                return True
        return False

    def finalize(self):
        if not self.done and self.count_map:
            best_ans = max(self.count_map, key=self.count_map.get)
            self.done = True
            self.final_answer = best_ans


class SelfConsistencyAggregator(AggregatorBase):
    """
    SC: Only consider first n_for_sc responses,
    pick the majority among them once we have n_for_sc.
    If we never get n_for_sc, fallback using what we have.
    """
    def __init__(self, n_for_sc=3):
        super().__init__("sc")
        self.n_for_sc = n_for_sc
        self.samples = []
        self.count_map = {}

    def add_sample(self, answer: str, conf: float):
        if self.done:
            return True
        self.used += 1
        self.samples.append(answer)
        self.count_map[answer] = self.count_map.get(answer, 0) + 1

        if self.used == self.n_for_sc:
            best_ans = max(self.count_map, key=self.count_map.get)
            self.final_answer = best_ans
            self.done = True
            return True
        return False

    def finalize(self):
        if not self.done and self.samples:
            best_ans = max(self.count_map, key=self.count_map.get)
            self.done = True
            self.final_answer = best_ans


class SelfConsistencyConfAggregator(AggregatorBase):
    """
    SC w/ conf: only consider first n_for_sc, pick the one with highest sum of conf.
    """
    def __init__(self, n_for_sc=3):
        super().__init__("sc_conf")
        self.n_for_sc = n_for_sc
        self.conf_map = {}
        self.count = 0

    def add_sample(self, answer: str, conf: float):
        if self.done:
            return True
        self.used += 1
        self.count += 1
        self.conf_map[answer] = self.conf_map.get(answer, 0.0)+conf

        if self.count == self.n_for_sc:
            best_ans = max(self.conf_map, key=self.conf_map.get)
            self.final_answer = best_ans
            self.done = True
            return True
        return False

    def finalize(self):
        if not self.done and self.conf_map:
            best_ans = max(self.conf_map, key=self.conf_map.get)
            self.done = True
            self.final_answer = best_ans


class BestOfNAggregator(AggregatorBase):
    """
    Best-of-n: among first n_for_bo, pick the single with highest conf.
    """
    def __init__(self, n_for_bo=3):
        super().__init__("best_of_n")
        self.n_for_bo = n_for_bo
        self.samples = []

    def add_sample(self, answer: str, conf: float):
        if self.done:
            return True
        self.used += 1
        self.samples.append((answer, conf))
        if self.used==self.n_for_bo:
            best_item = max(self.samples, key=lambda x: x[1])
            self.final_answer = best_item[0]
            self.done = True
            return True
        return False

    def finalize(self):
        if not self.done and self.samples:
            best_item = max(self.samples, key=lambda x: x[1])
            self.final_answer = best_item[0]
            self.done = True


def aggregator_factory(method: str, threshold=0.8, n_for_sc=3):
    method = method.lower()
    if method == "earlyexit":
        return EarlyStopAggregator(threshold=threshold)
    elif method == "asc_conf":
        return AscConfAggregator(threshold=threshold)
    elif method == "asc":
        return AscAggregator(threshold=threshold)
    elif method == "sc":
        return SelfConsistencyAggregator(n_for_sc=n_for_sc)
    elif method == "sc_conf":
        return SelfConsistencyConfAggregator(n_for_sc=n_for_sc)
    elif method == "best_of_n":
        return BestOfNAggregator(n_for_bo=n_for_sc)
    else:
        raise ValueError(f"Unknown method={method}")


class SampleInference:
    def __init__(
        self,
        model_name: str,
        eos_token_str: str,
        I: str,
        torch_dtype=torch.float16,
        device_map="auto"
    ):
        self.model_name = model_name
        print(f"Loading model from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        self.model.eval()

        # Encode the fixed prompt I
        self.I_text = I
        self.I_ids = self.tokenizer.encode(I, add_special_tokens=False, return_tensors='pt').to(self.model.device)

        # parse eos token
        eos_ids = self.tokenizer.encode(eos_token_str, add_special_tokens=False)
        if eos_ids:
            self.eos_token_id = eos_ids[0]
        else:
            self.eos_token_id = None
        print("Initialization done.")

    def generate_one_response(
        self,
        query: str,
        temperature: float,
        max_new_tokens: int,
        need_conf: bool
    ):
        """
        1) Generate a single text 'answer_text' from 'query'.
        2) If need_conf is True, then compute 'prob_yes' by appending self.I
           else, skip it (set prob_yes=0.0)
        Return (answer_text, prob_yes).
        """
        # ========== 1) Encode query, do forward + sampling ==========

        query_ids = self.tokenizer.encode(query, add_special_tokens=False, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            out_query = self.model(query_ids, use_cache=True)
        pkv = out_query.past_key_values

        generated_ids = []
        for _ in range(max_new_tokens):
            if len(generated_ids)==0:
                input_ids = query_ids[:, -1:]
            else:
                input_ids = torch.tensor([[generated_ids[-1]]], dtype=torch.long).to(self.model.device)

            with torch.no_grad():
                out = self.model(input_ids, past_key_values=pkv, use_cache=True)
            logits = out.logits[:, -1, :]
            pkv = out.past_key_values

            if temperature!=1.0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

            next_id_t = torch.multinomial(probs, num_samples=1)
            next_id = int(next_id_t[0,0].item())

            if self.eos_token_id is not None and next_id == self.eos_token_id:
                break
            generated_ids.append(next_id)

        answer_text = self.tokenizer.decode(generated_ids)

        # ========== 2) If aggregator needs confidence, compute prob_yes by appending I ==========
        if need_conf:
            with torch.no_grad():
                out_I = self.model(self.I_ids, past_key_values=pkv, use_cache=True)
            logits_last = out_I.logits[:, -1, :]
            p = torch.softmax(logits_last, dim=-1)
            yes_ids = self.tokenizer.encode("Yes", add_special_tokens=False)
            if yes_ids:
                yes_id = yes_ids[0]
                prob_yes = float(p[0, yes_id].item())
            else:
                prob_yes = 0.0
        else:
            prob_yes = 0.0

        return answer_text, prob_yes

    def run_inference_interactive(
        self,
        query: str,
        method: str,
        threshold: float,
        max_samples: int,
        temperature: float,
        extract_handler
    ):

        agg = aggregator_factory(method, threshold, max_samples)
        methods_need_conf = ["earlyexit", "asc_conf", "sc_conf", "best_of_n"]
        need_conf = (agg.name in methods_need_conf)

        all_samples = []
        for i in range(max_samples):
            # 1) Generate one response
            answer_text, conf = self.generate_one_response(
                query=query,
                temperature=temperature,
                max_new_tokens=1024,
                need_conf=need_conf
            )

            # 2) Extract the 'parsed answer' using handler
            parsed_ans = extract_handler.extract_answer(answer_text)

            # 3) aggregator add
            stopped = agg.add_sample(parsed_ans, conf)

            # 4) record
            all_samples.append((answer_text, parsed_ans, conf))

            if stopped:
                break

        if not agg.done:
            agg.finalize()

        return {
            "method": method,
            "final_answer": agg.final_answer if agg.final_answer is not None else parsed_ans,
            "used_samples": agg.used,
            "all_samples": all_samples
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cot", action="store_true", help="Use chain-of-thought format.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    args = parser.parse_args()


    model_name = args.model_name
    I = get_suffix(model_name)
    eos_token_str = get_eos_token(model_name)

    inference = SampleInference(
        model_name=model_name,
        eos_token_str=eos_token_str,
        I=I,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    dataset_handler = get_dataset(args.dataset_name)

    data, answer_type = dataset_handler.load_data()
    data = data['test'].select(range(10))
    qa_data = dataset_handler.prepare_qa_data(data)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts, qa_data = generate_prompt(args, logger, qa_data, answer_type=answer_type, tokenizer=tokenizer)
    aggregator_list = ["earlyexit", "asc_conf", "asc", "sc", "sc_conf", "best_of_n"]
    results = {}  # to store accuracy, usage, time for each method

    for method in tqdm(aggregator_list):
        start_time = time.time()
        acc = 0
        response_usage = 0
        num_samples = len(prompts)  # e.g. 100 if you selected data[:100]

        for prompt, answer in tzip(prompts, list(qa_data.values())):
            # 1) Run inference with the chosen aggregator method
            result = inference.run_inference_interactive(
                query=prompt,
                method=method,
                threshold=0.51,
                max_samples=16,
                temperature=0.8,
                extract_handler=dataset_handler
            )
            
            # 2) Check if final answer is correct
            parsed_ground_truth = dataset_handler.extract_answer(answer)
            if result['final_answer'] == parsed_ground_truth:
                acc += 1
            
            # 3) Accumulate how many responses were used for this question
            response_usage += result['used_samples']

        # measure elapsed time
        elapsed_time = time.time() - start_time

        # store results for this aggregator
        results[method] = {
            "accuracy": acc / num_samples,
            "avg_usage": response_usage / num_samples,
            "time_sec": elapsed_time
        }

    # Print summary of all methods
    print("\n=== Comparison of Aggregator Methods ===")
    for method in aggregator_list:
        print(f"Method: {method}")
        print(f"  Accuracy:       {results[method]['accuracy']:.3f}")
        print(f"  Avg Usage:      {results[method]['avg_usage']:.2f}")
        print(f"  Elapsed (sec):  {results[method]['time_sec']:.2f}")
        print()


if __name__ == "__main__":
    main()
