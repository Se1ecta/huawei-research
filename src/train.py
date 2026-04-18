import argparse
import math
import os
from enum import Enum

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)

from data_utils import pack_dataset
from optimizers.muon import Muon
from utils import set_seed


class OptimizerNames(str, Enum):
    adamw = "adamw"
    muon = "muon"


def log_peak_memory() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def build_dataset(dataset_name, tokenizer, seq_len):

    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
    }

    dataset = load_dataset(name2path[dataset_name], split="train")

    dataset = dataset.select(range(10_000))

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=False,
        )

    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        num_proc=4,
    )

    dataset = pack_dataset(
        dataset,
        seq_length=seq_len,
        strategy="bfd_split",
    )

    return dataset


def build_optimizer(name: str, model, lr: float, wd: float):

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=(0.9, 0.95),
        )

    muon_params = []
    adam_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if name == "muon":
            if (
                p.ndim == 2
                and "embed_tokens" not in n
                and "lm_head" not in n
                and "norm" not in n
            ):
                muon_params.append(p)
            else:
                adam_params.append(p)

        elif name == "hybrid":
            if "attn" in n and p.ndim == 2:
                muon_params.append(p)
            else:
                adam_params.append(p)

    return Muon(
        muon_params=muon_params,
        adamw_params=adam_params,
        lr=lr,
        wd=wd,
    )


def train(args):

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if args.fp32 else torch.float16,
        trust_remote_code=True,
    )

    train_dataset = build_dataset(
        args.dataset,
        tokenizer,
        args.seq_len,
    )

    total_steps = (
        math.ceil(len(train_dataset) / args.batch_size / args.grad_accum) * args.epochs
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.wd,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=args.log_every,
        save_strategy="no",
        fp16=not args.fp32,
        bf16=False,
        report_to=["tensorboard"],
        seed=args.seed,
        dataloader_pin_memory=True,
    )

    optimizer = build_optimizer(
        args.optimizer,
        model,
        lr=args.lr,
        wd=args.wd,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.03 * total_steps),
        num_training_steps=total_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        optimizers=(optimizer, scheduler),
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--dataset", default="openwebtext-100k")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=512)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=0.01)

    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)

    parser.add_argument(
        "--output_dir", default="/content/drive/MyDrive/Huawei-research"
    )

    args = parser.parse_args()

    train(args)
