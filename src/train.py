"""Оснвовной скрипт обучения."""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import torch
from clearml import Task
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from mezo.mezo_trainer import MeZoTrainer
from optimizers.muon import Muon


class OptimizerNames(str, Enum):
    """Названия оптимизаторов используемых в экспериментах."""

    adamw = "adamw"
    muon = "muon"
    hybrid_muon = "hybrid_muon"
    mezo = "mezo"


@dataclass
class ScriptArguments:
    """Аргументы, специфичные для скрипта."""

    optimizer: OptimizerNames = field(
        default=OptimizerNames.adamw, metadata={"help": "Тип оптимизатора"}
    )


@dataclass
class ModelArguments:
    """Аргументы, связанные с моделью и токенизатором."""

    model_name: str = field(
        default="Qwen/Qwen2.5-0.5B",
        metadata={"help": "Имя модели или путь к ней для загрузки токенизатора"},
    )


@dataclass
class DataArguments:
    """Аргументы, связанные с данными."""

    dataset_name: str = field(
        default="openwebtext-100k",
        metadata={"help": "Имя датасета из списка предопределённых"},
    )
    train_subset_size: Optional[int] = field(
        default=10_000,
        metadata={"help": "Количество примеров для обучения (None = все)"},
    )
    seq_length: int = field(
        default=512, metadata={"help": "Длина последовательности для группировки"}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Нужен для переопределения дефолтных аргументов в остальном тот же TrainingArguments"""

    warmup_ratio: Optional[float] = field(
        default=0.0,
        metadata={"help": "Warmap ratio"},
    )

    zo_eps: Optional[float] = field(
        default=1e-3,
        metadata={"help": "eps in MeZO"},
    )


def setup_logging() -> None:
    """Настраивает базовое логирование."""
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )


def build_dataset(data_args: DataArguments, tokenizer) -> torch.utils.data.Dataset:
    """Загружает и подготавливает датасет для языкового моделирования."""
    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
    }
    if data_args.dataset_name not in name2path:
        raise ValueError(
            f"Неизвестный датасет: {data_args.dataset_name}. Доступны: {list(name2path.keys())}"
        )

    dataset = load_dataset(name2path[data_args.dataset_name], split="train")

    if data_args.train_subset_size:
        dataset = dataset.select(range(data_args.train_subset_size))

    def tokenize(batch):
        return tokenizer(batch["text"])

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // data_args.seq_length) * data_args.seq_length

        result = {
            k: [
                concatenated[k][i : i + data_args.seq_length]
                for i in range(0, total_length, data_args.seq_length)
            ]
            for k in concatenated.keys()
        }

        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(group_texts, batched=True)

    return dataset


def build_muon_optimizer(
    name: str, model, lr: float, wd: float
) -> torch.optim.Optimizer:
    """Фабричный метод для получения muon оптимизатора."""

    muon_params = []
    adam_params = []

    for n, p in model.named_parameters():
        if name == OptimizerNames.muon:
            if (
                p.ndim == 2
                and "embed_tokens" not in n
                and "lm_head" not in n
                and "norm" not in n
            ):
                muon_params.append(p)
            else:
                adam_params.append(p)

        elif name == OptimizerNames.hybrid_muon:
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


class TrainerWithMuonOptimizer(Trainer):
    """Trainer для работы с Muon."""

    def __init__(self, optimizer_name, *args, **kwargs):
        self.optimizer_name = optimizer_name
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_optimizer(self) -> torch.optim.Optimizer:
        self.optimizer = build_muon_optimizer(
            self.optimizer_name,
            self.model,
            lr=self.args.learning_rate,
            wd=self.args.weight_decay,
        )
        self.logger.info(
            f"✅ Используется оптимизатор: {type(self.optimizer).__name__}"
        )
        return self.optimizer


def build_task_name(
    optimizer_name: OptimizerNames,
    training_args: TrainingArguments,
    data_args: DataArguments,
) -> str:
    """Формирует имя задачи для ClearML."""
    date = datetime.now().strftime("%m%d_%H%M")
    return (
        f"{optimizer_name}"
        f"_lr{training_args.learning_rate:.1e}"
        f"_bs{training_args.per_device_train_batch_size}"
        f"_wd{training_args.weight_decay}"
        f"_sl{data_args.seq_length}"
        f"_{date}"
    )


def init_clearml_task(
    optimizer_name: OptimizerNames,
    training_args: TrainingArguments,
    data_args: DataArguments,
) -> Task:
    """Инициализирует задачу ClearML."""
    task = Task.init(
        project_name="Huawei-research",
        task_name=build_task_name(optimizer_name, training_args, data_args),
        tags=[
            optimizer_name,
            f"lr={training_args.learning_rate:.1e}",
            f"bs={training_args.per_device_train_batch_size}",
        ],
    )
    return task


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    script_args: ScriptArguments,
    training_args: TrainingArguments,
):

    init_clearml_task(script_args.optimizer, training_args, data_args)

    set_seed(training_args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(training_args.output_dir, exist_ok=True)  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name).to(device)  # type: ignore

    train_dataset = build_dataset(data_args, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if script_args.optimizer == OptimizerNames.mezo:
        trainer = MeZoTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
    elif (
        script_args.optimizer == OptimizerNames.muon
        or script_args.optimizer == OptimizerNames.hybrid_muon
    ):
        trainer = TrainerWithMuonOptimizer(
            optimizer_name=script_args.optimizer,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = HfArgumentParser(
        [ModelArguments, DataArguments, ScriptArguments, CustomTrainingArguments]
    )

    model_args, data_args, script_args, training_args = (
        parser.parse_args_into_dataclasses()
    )

    train(model_args, data_args, script_args, training_args)
