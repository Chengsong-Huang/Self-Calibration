import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))


import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from huggingface_hub import login
import math
from collections import defaultdict
from utils.SPECIAL_SUFFIXS import *
# ============================================================================
# Argument Parser
# ============================================================================
parser = argparse.ArgumentParser(description="Multi-task training with LoRA & aggregated loss logging.")
parser.add_argument("--config_file", required=True, help="Path to the configuration file.")
parser.add_argument("--save_path", type=str)
args = parser.parse_args()

# ============================================================================
# Load config
# ============================================================================
with open(args.config_file, "r") as f:
    config = json.load(f)
if args.save_path:
    save_path = args.save_path
else:save_path = config["output_dir"]
# ============================================================================
# (Optional) Hugging Face login if needed
# ============================================================================
# login(config["hf_token"])

# ============================================================================
# Initialize distributed training if required
# ============================================================================
if config["distributed"]:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
else:
    local_rank = 0

# ============================================================================
# Initialize Weights & Biases
# ============================================================================
if local_rank == 0 and config.get("use_wandb", False):
    import wandb
    wandb.init(project=config["wandb_project"])

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# Hyperparameters from config
# ============================================================================
BATCH_SIZE = config["batch_size"]
GRADIENT_ACCUMULATION_STEPS = config["gradient_accumulation_steps"]
NUM_EPOCHS = config["num_epochs"]
causal_lm_ratio = config["causal_lm_ratio"]
total_train_samples = config["total_train_samples"]
total_eval_samples = config["total_eval_samples"]
d_n = config["dataset"]
LOG_INTERVAL = config.get("log_interval", 10)  # default to 10 if not provided

SPECIAL_SUFFIX = get_suffix(config["model_name"])
# ============================================================================
# Tokenizer initialization
# ============================================================================
tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# "Yes" token ID for Weighted task
token_yes = tokenizer.convert_tokens_to_ids("Yes")
split_marker = get_split(config["model_name"].lower())
print("Yes id:",token_yes)
print('split marker',split_marker)
# ============================================================================
# Data processing functions
# ============================================================================

def add_special_suffix(example):
    """
    Append SPECIAL_SUFFIX and an EOS token at the end of the example.
    This helps the model see a consistent end-of-sequence marker.
    """
    example["input"] += f" {SPECIAL_SUFFIX}"
    return example

def process_causal_lm(example, dataset_name=None):
    """
    For a (user + assistant) conversation:
    1) Split at <|start_header_id|>assistant<|end_header_id|> to separate user & assistant text.
    2) Only the assistant part is used for cross-entropy (the user part is masked out with -100).
    3) We'll pass 'labels' directly to model(..., labels=...), letting it do the correct shift.
    """
    parts = example["input"].split(split_marker)
    if len(parts) == 2:
        user_part = parts[0] + split_marker
        assistant_part = parts[1].strip() + tokenizer.eos_token

        full_input = user_part + assistant_part
        encoded = tokenizer(
            full_input, 
            truncation=True, 
            max_length=1024, 
            padding='max_length', 
            return_tensors='pt'
        )
        input_ids = encoded.input_ids.squeeze(0)
        labels = input_ids.clone()

        user_len = len(tokenizer(user_part, truncation=True, max_length=1024)["input_ids"])
        labels[:user_len] = -100

        return {
            "task": "causal_lm",
            "input_ids": input_ids,
            "labels": labels,
            "weighted_target": -1.0,
            "dataset_name": dataset_name
        }
    else:
        return print(f"Error: split marker not found in example: {example}")

def process_weighted(example, dataset_name=None):
    """
    For 'weighted' consistency:
    1) Append the special suffix + EOS token so the model sees the full prompt including "Is the answer correct?"
    2) We do NOT compute cross-entropy. Labels are all set to -100.
    """
    example = add_special_suffix(example)
    wc = example["weighted_consistency"]
    if wc is None or np.isnan(wc):
        return None
    wc = max(0.0, min(1.0, float(wc)))

    encoded = tokenizer(
        example["input"], 
        truncation=True, 
        max_length=1024, 
        padding='max_length', 
        return_tensors='pt'
    )
    input_ids = encoded.input_ids.squeeze(0)
    labels = torch.full_like(input_ids, -100)

    return {
        "task": "weighted",
        "input_ids": input_ids,
        "labels": labels,
        "weighted_target": wc,
        "dataset_name": dataset_name
    }

def add_dataset_name(example, dataset_name):
    example["dataset_name"] = dataset_name
    return example

# ============================================================================
# Load Datasets
# ============================================================================
datasets_list = []
for dataset_info in config["datasets"]:
    dataset_name = dataset_info["name"]
    loaded = load_dataset(d_n, dataset_name)
    if local_rank == 0:
        print(f"Loaded dataset: {dataset_name}, splits: {loaded.keys()}")
    datasets_list.append((dataset_name, loaded))

mixed_train_data = []
mixed_eval_data = []

# We'll define "slots" for bucketing weighted_consistency
slots = np.arange(0, 1.05, 0.05)

for dataset_info, (dataset_name, dataset) in zip(config["datasets"], datasets_list):
    train_percentage = dataset_info["train_percentage"]
    eval_percentage = dataset_info["eval_percentage"]

    if local_rank == 0:
        print(f"Processing dataset {dataset_name}...")

    # ================
    # Training portion
    # ================
    train_dataset = dataset["train"].filter(
        lambda x: x["weighted_consistency"] is not None
                  and not np.isnan(x["weighted_consistency"])
                  and x["answer"] is not None
    )

    train_dataset = train_dataset.map(lambda x: add_dataset_name(x, dataset_name))

    causal_lm_data = train_dataset.filter(lambda x: x["weighted_consistency"] > config["threshold"])
    causal_lm_data = causal_lm_data.map(lambda x: process_causal_lm(x, dataset_name=dataset_name))
    causal_lm_data = causal_lm_data.filter(lambda x: x is not None)
    causal_lm_list = list(causal_lm_data)

    # Bucket sampling for weighted
    slot_groups = {i: [] for i in range(len(slots))}
    for ex in train_dataset:
        val = ex["weighted_consistency"]
        if val is not None and not np.isnan(val):
            slot_idx = int(min(val // 0.05, len(slots) - 1))
            slot_groups[slot_idx].append(ex)

    weighted_samples = []
    samples_per_slot = int((total_train_samples * (1 - causal_lm_ratio) * train_percentage) // len(slots))
    for slot_idx, slot_ex_list in slot_groups.items():
        if len(slot_ex_list) > 0:
            picked = random.sample(slot_ex_list, min(len(slot_ex_list), samples_per_slot))
            for eex in picked:
                ww = process_weighted(eex, dataset_name=dataset_name)
                if ww is not None:
                    weighted_samples.append(ww)

    # Now sample from the causal_lm_list
    num_causal_lm_samples = int(total_train_samples * causal_lm_ratio * train_percentage)
    causal_samples = random.sample(causal_lm_list, min(len(causal_lm_list), num_causal_lm_samples))

    random.shuffle(weighted_samples)
    random.shuffle(causal_samples)
    combined_train = weighted_samples + causal_samples
    mixed_train_data.extend(combined_train)

    # ================
    # Evaluation portion
    # ================
    eval_dataset = dataset["test"].filter(
        lambda x: x["weighted_consistency"] is not None
                  and not np.isnan(x["weighted_consistency"])
                  and x["answer"] is not None
    )
    eval_dataset = eval_dataset.map(lambda x: add_dataset_name(x, dataset_name))

    causal_lm_eval = eval_dataset.filter(lambda x: x["weighted_consistency"] > 0.75)
    causal_lm_eval = causal_lm_eval.map(lambda x: process_causal_lm(x, dataset_name=dataset_name))
    causal_lm_eval = causal_lm_eval.filter(lambda x: x is not None)
    causal_lm_eval_list = list(causal_lm_eval)

    slot_groups_eval = {i: [] for i in range(len(slots))}
    for ex in eval_dataset:
        val = ex["weighted_consistency"]
        if val is not None and not np.isnan(val):
            si = int(min(val // 0.05, len(slots)-1))
            slot_groups_eval[si].append(ex)

    weighted_eval_samples = []
    samples_per_slot_eval = int((total_eval_samples * (1 - causal_lm_ratio) * eval_percentage) // len(slots))
    for slot_idx, slot_ex_list in slot_groups_eval.items():
        if len(slot_ex_list) > 0:
            picked_eval = random.sample(slot_ex_list, min(len(slot_ex_list), samples_per_slot_eval))
            for eex2 in picked_eval:
                rr = process_weighted(eex2, dataset_name=dataset_name)
                if rr is not None:
                    weighted_eval_samples.append(rr)

    num_causal_lm_eval_samples = int(total_eval_samples * causal_lm_ratio * eval_percentage)
    causal_eval_samples = random.sample(causal_lm_eval_list, min(len(causal_lm_eval_list), num_causal_lm_eval_samples))

    random.shuffle(weighted_eval_samples)
    random.shuffle(causal_eval_samples)
    combined_eval = weighted_eval_samples + causal_eval_samples
    mixed_eval_data.extend(combined_eval)

# Shuffle final merged data
random.shuffle(mixed_train_data)
random.shuffle(mixed_eval_data)

if local_rank == 0:
    print("Final mixed_train_data size:", len(mixed_train_data))
    print("Final mixed_eval_data size:", len(mixed_eval_data))

# ============================================================================
# Build PyTorch Datasets
# ============================================================================
class MixedDataset(TorchDataset):
    """
    A dataset holding both causal_lm and weighted samples in a single list.
    Each item has structure: {task, input_ids, labels, weighted_target, dataset_name}.
    """
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        ex = self.data_list[idx]
        task = ex["task"]
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        weighted_target = ex["weighted_target"]
        dataset_name = ex.get("dataset_name", "unknown")

        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        if not isinstance(weighted_target, float):
            weighted_target = float(weighted_target)
        weighted_target = torch.tensor(weighted_target, dtype=torch.float)

        return (task, input_ids, labels, weighted_target, dataset_name)

train_dataset = MixedDataset(mixed_train_data)
eval_dataset  = MixedDataset(mixed_eval_data)

# ============================================================================
# Create DataLoaders (with optional distributed sampler)
# ============================================================================
if config["distributed"]:
    from torch.utils.data.distributed import DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
else:
    train_sampler = None
    eval_sampler = None

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    drop_last=True
)

eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    sampler=eval_sampler,
    drop_last=True
)

# ============================================================================
# Model initialization & LoRA
# ============================================================================
model = AutoModelForCausalLM.from_pretrained(config["model_name"])
model.resize_token_embeddings(len(tokenizer))
try:
    lora_cfg = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["q_proj", "v_proj"],
        lora_dropout=config["lora_dropout"],
        bias="none"
    )
    model = get_peft_model(model, lora_cfg)
except:
    lora_cfg = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["qkv_proj", "o_proj"],
        lora_dropout=config["lora_dropout"],
        bias="none"
    )
    model = get_peft_model(model, lora_cfg)
model.to(local_rank)

if config["distributed"]:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# ============================================================================
# Optimizer & Scheduler
# ============================================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
steps_per_epoch = len(train_dataloader)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * steps_per_epoch)

# MSE for Weighted
mse_criterion = nn.SmoothL1Loss(beta=0.25)


# ============================================================================
# Modified evaluation function: separate by dataset_name
# ============================================================================
def evaluate(model, eval_loader, local_rank):
    """
    Evaluate on both tasks, but also separate by dataset_name.
    Return a dict like:
    {
      'dataset_name_1': {'avg_ce_loss': x, 'avg_mse_loss': y},
      'dataset_name_2': {...},
      ...
    }
    """
    model.eval()
    dataset_stats = defaultdict(lambda: {
        "ce_sum": 0.0,
        "ce_count": 0,
        "mse_sum": 0.0,
        "mse_count": 0
    })

    with torch.no_grad():
        for tasks, input_ids, labels, weighted_targets, dataset_names in eval_loader:
            input_ids = input_ids.to(local_rank)
            labels = labels.to(local_rank)
            weighted_targets = weighted_targets.to(local_rank)

            batch_size = len(tasks)
            for i in range(batch_size):
                task = tasks[i]
                ds_name = dataset_names[i]

                single_input_ids = input_ids[i].unsqueeze(0)     # [1, seq_len]
                single_labels = labels[i].unsqueeze(0)           # [1, seq_len]
                single_weighted_target = weighted_targets[i].unsqueeze(0)

                if task == "causal_lm":
                    out_causal = model(input_ids=single_input_ids, labels=single_labels)
                    loss_causal = out_causal.loss.item()
                    dataset_stats[ds_name]["ce_sum"] += loss_causal
                    dataset_stats[ds_name]["ce_count"] += 1

                elif task == "weighted":
                    out_weighted = model(input_ids=single_input_ids)
                    logits_weighted = out_weighted.logits  # [1, seq_len, vocab_size]
                    last_logits = logits_weighted[:, -1, :]
                    yes_prob = F.softmax(last_logits, dim=-1)[:, token_yes]
                    loss_mse = mse_criterion(yes_prob, single_weighted_target)
                    dataset_stats[ds_name]["mse_sum"] += loss_mse.item()
                    dataset_stats[ds_name]["mse_count"] += 1

    model.train()

    results = {}
    for ds_name, stat in dataset_stats.items():
        ce_count = stat["ce_count"]
        mse_count = stat["mse_count"]
        avg_ce = stat["ce_sum"] / ce_count if ce_count > 0 else float('nan')
        avg_mse = stat["mse_sum"] / mse_count if mse_count > 0 else float('nan')
        results[ds_name] = {
            "avg_ce_loss": avg_ce,
            "avg_mse_loss": avg_mse
        }

    return results

# ============================================================================
# Training loop with aggregated loss logging
# ============================================================================
global_step = 0
model.train()

acc_loss_causal = 0.0
acc_loss_weighted = 0.0
count_causal_batches = 0
count_weighted_batches = 0

for epoch in range(NUM_EPOCHS):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", disable=(local_rank != 0))
    optimizer.zero_grad()

    for batch_idx, batch_data in enumerate(pbar, start=1):
        tasks, input_ids, labels, weighted_targets, dataset_names = batch_data
        input_ids = input_ids.to(local_rank)
        labels = labels.to(local_rank)
        weighted_targets = weighted_targets.to(local_rank)

        causal_mask = [(t == "causal_lm") for t in tasks]
        causal_mask = torch.tensor(causal_mask, dtype=torch.bool, device=local_rank)

        weighted_mask = [(t == "weighted") for t in tasks]
        weighted_mask = torch.tensor(weighted_mask, dtype=torch.bool, device=local_rank)

        loss_causal = torch.tensor(0.0, device=local_rank)
        loss_weighted = torch.tensor(0.0, device=local_rank)

        # ---------- CAUSAL LM part ----------
        if causal_mask.any():
            inp_causal = input_ids[causal_mask]
            lab_causal = labels[causal_mask]
            out_causal = model(input_ids=inp_causal, labels=lab_causal)
            raw_loss_causal = out_causal.loss
            loss_causal = raw_loss_causal / 10.0  # optional scaling
            count_causal_batches += 1

        # ---------- WEIGHTED part ----------
        if weighted_mask.any():
            inp_weighted = input_ids[weighted_mask]
            tgt_weighted = weighted_targets[weighted_mask]
            out_weighted = model(input_ids=inp_weighted)
            logits_weighted = out_weighted.logits  # [N, seq_len, vocab_size]
            last_logits = logits_weighted[:, -1, :]
            yes_prob = F.softmax(last_logits, dim=-1)[:, token_yes]
            valid_mask = (tgt_weighted >= 0)
            if valid_mask.sum() > 0:
                raw_loss_weighted = mse_criterion(yes_prob[valid_mask], tgt_weighted[valid_mask])
            else:
                raw_loss_weighted = torch.tensor(0.0, device=local_rank)

            loss_weighted = raw_loss_weighted
            count_weighted_batches += 1

        # Combine for backward
        loss = (loss_causal + loss_weighted) / GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        global_step += 1
        acc_loss_causal += loss_causal.item()
        acc_loss_weighted += loss_weighted.item()

        # Show combined step loss in progress bar
        combined_step_loss = loss_causal.item() + loss_weighted.item()
        if local_rank == 0:
            pbar.set_postfix({"loss": combined_step_loss})

        # Gradient accumulation
        if global_step % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ------------- Log at defined interval -------------
        if (global_step % LOG_INTERVAL == 0) and (local_rank == 0) and config.get("use_wandb", False):
            if count_causal_batches > 0:
                avg_causal_loss = acc_loss_causal / count_causal_batches
            else:
                avg_causal_loss = 0.0

            if count_weighted_batches > 0:
                avg_weighted_loss = acc_loss_weighted / count_weighted_batches
            else:
                avg_weighted_loss = 0.0

            wandb.log({
                "step": global_step,
                "train_loss_causal_avg": avg_causal_loss,
                "train_loss_weighted_avg": avg_weighted_loss
            })

            acc_loss_causal = 0.0
            acc_loss_weighted = 0.0
            count_causal_batches = 0
            count_weighted_batches = 0

        # ------------- Optional intermediate evaluation -------------
        if config.get("use_wandb", False) and global_step % 500 == 0:
            eval_results = evaluate(model, eval_dataloader, local_rank)
            if  local_rank == 0 and config.get("use_wandb", False):
                for ds_name, metrics in eval_results.items():
                    wandb.log({
                        f"{ds_name}_ce_loss": metrics["avg_ce_loss"],
                        f"{ds_name}_mse_loss": metrics["avg_mse_loss"],
                        "step": global_step
                    })
                print(f"[Epoch {epoch+1}] Per-dataset eval results: {eval_results}")
                checkpoint_dir = os.path.join(save_path, f"checkpoint_step_{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)

                final_model = model.module if hasattr(model, 'module') else model
                final_model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)

    pbar.close()


# ============================================================================
# Save final model
# ============================================================================
if local_rank == 0:
    os.makedirs(save_path, exist_ok=True)
    final_model = model.module if hasattr(model, 'module') else model
    final_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
