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
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Qwen2Config,
    Qwen2ForCausalLM,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    set_seed,
)

from mezo.mezo_trainer import MeZoTrainer
from optimizers.muon import Muon


class OptimizerNames(str, Enum):
    adamw = "adamw"
    muon = "muon"
    hybrid_muon = "hybrid_muon"
    mezo = "mezo"


@dataclass
class ModelArguments:
    """Аргументы, связанные с моделью и токенизатором."""

    model_name: str = field(
        default="Qwen/Qwen2.5-0.5B",
        metadata={"help": "Имя модели или путь к ней для загрузки токенизатора"},
    )
    hidden_size: int = field(
        default=1024, metadata={"help": "Размер скрытого состояния"}
    )
    intermediate_size: int = field(
        default=4864, metadata={"help": "Размер промежуточного слоя FFN"}
    )
    num_hidden_layers: int = field(
        default=12, metadata={"help": "Количество слоёв Transformer"}
    )
    num_attention_heads: int = field(
        default=16, metadata={"help": "Количество голов внимания"}
    )
    num_key_value_heads: int = field(
        default=16, metadata={"help": "Количество голов для KV-кэша"}
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
class OptimizerArguments:
    """Аргументы оптимизатора и обучения."""

    optimizer: OptimizerNames = field(
        default=OptimizerNames.adamw, metadata={"help": "Тип оптимизатора"}
    )
    learning_rate: float = field(
        default=5e-5, metadata={"help": "Начальная скорость обучения"}
    )
    weight_decay: float = field(
        default=0.01, metadata={"help": "Коэффициент регуляризации L2"}
    )
    warmup_ratio: float = field(default=0.03, metadata={"help": "Доля warmup-шагов"})
    batch_size: int = field(default=2, metadata={"help": "Размер батча на устройство"})
    grad_accumulation_steps: int = field(
        default=8, metadata={"help": "Шаги накопления градиента"}
    )
    num_epochs: int = field(default=1, metadata={"help": "Количество эпох обучения"})
    use_fp16: bool = field(
        default=True, metadata={"help": "Использовать mixed precision (FP16)"}
    )
    seed: int = field(default=42, metadata={"help": "Сид для воспроизводимости"})
    log_every_steps: int = field(
        default=10, metadata={"help": "Логировать каждые N шагов"}
    )
    output_dir: str = field(
        default="./outputs", metadata={"help": "Директория для сохранения"}
    )


def setup_logging() -> None:
    """Настраивает базовое логирование."""
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )


def build_dataset(dataset_name, tokenizer, seq_len):

    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
    }

    dataset = load_dataset(name2path[dataset_name], split="train")

    dataset = dataset.select(range(10_000))

    def tokenize(batch):
        return tokenizer(batch["text"])

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // seq_len) * seq_len

        result = {
            k: [
                concatenated[k][i : i + seq_len]
                for i in range(0, total_length, seq_len)
            ]
            for k in concatenated.keys()
        }

        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(group_texts, batched=True)

    return dataset


def build_optimizer(name: str, model, lr: float, wd: float) -> torch.optim.Optimizer:

    if name == OptimizerNames.adamw:
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=(0.9, 0.95),
        )

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


def build_model_config(
    model_args: ModelArguments, tokenizer, max_seq_len: int
) -> Qwen2Config:
    """Создаёт конфигурацию модели Qwen2."""
    return Qwen2Config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=model_args.hidden_size,
        intermediate_size=model_args.intermediate_size,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=model_args.num_attention_heads,
        num_key_value_heads=model_args.num_key_value_heads,
        max_position_embeddings=max_seq_len + 1,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )


class TrainerWithMuonOptimizer(Trainer):
    def __init__(self, optimizer_name: str, *args, **kwargs):
        self.optimizer_name = optimizer_name
        super().__init__(*args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):

        if self.optimizer_name == OptimizerNames.mezo:
            return  # MeZO uses its own trainer

        optimizer = build_optimizer(
            self.optimizer_name,
            self.model,
            self.args.learning_rate,
            self.args.weight_decay,
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.args.warmup_ratio * num_training_steps),
            num_training_steps=num_training_steps,
        )

        self.optimizer = optimizer
        self.lr_scheduler = scheduler


def build_task_name(args):
    date = datetime.now().strftime("%m%d_%H%M")
    return (
        f"{args.optimizer}"
        f"_lr{args.lr:.1e}"
        f"_bs{args.batch_size}"
        f"_wd{args.wd}"
        f"_sl{args.seq_len}"
        f"_{date}"
    )


def init_clearml_task(args):
    task = Task.init(
        project_name="Huawei-research",
        task_name=build_task_name(args),
        tags=[
            args.optimizer,
            f"lr={args.lr:.1e}",
            f"bs={args.batch_size}",
        ],
    )
    return task


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    opt_args: OptimizerArguments,
    training_args: TrainingArguments,
):

    init_clearml_task(opt_args, data_args)

    set_seed(opt_args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt_args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = build_model_config(model_args, tokenizer, data_args.seq_length)
    model = Qwen2ForCausalLM(config).to(device)

    model = Qwen2ForCausalLM(config).to(device)

    train_dataset = build_dataset(data_args, tokenizer, data_args.seq_length)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=opt_args.output_dir,
        per_device_train_batch_size=opt_args.batch_size,
        gradient_accumulation_steps=opt_args.grad_accumulation_steps,
        num_train_epochs=opt_args.num_epochs,
        learning_rate=opt_args.learning_rate,
        weight_decay=opt_args.weight_decay,
        warmup_ratio=opt_args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=opt_args.log_every_steps,
        save_strategy="epoch",
        fp16=opt_args.use_fp16 and torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to=["tensorboard", "clearml"],
        remove_unused_columns=False,
        **{k: v for k, v in training_args.to_dict().items() if v is not None},
    )

    if opt_args.optimizer == OptimizerNames.mezo:
        trainer = MeZoTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
    else:
        trainer = TrainerWithMuonOptimizer(
            optimizer_name=opt_args.optimizer,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

    trainer.train()
    trainer.save_model(opt_args.output_dir)
    tokenizer.save_pretrained(opt_args.output_dir)


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = HfArgumentParser(
        [ModelArguments, DataArguments, OptimizerArguments, TrainingArguments]
    )
    try:
        model_args, data_args, opt_args, training_args = (
            parser.parse_args_into_dataclasses()
        )
    except SystemExit:
        parser.print_help()
        exit(0)

    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"Optimizer args: {opt_args}")
    logger.info(f"Training args: {training_args}")

    train(model_args, data_args, opt_args, training_args)
