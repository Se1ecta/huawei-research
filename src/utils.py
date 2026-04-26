"""Утилиты используемые в проекте."""

import logging
import os
import random

import torch


def set_seed(seed: int) -> None:
    """Установка random seed в различные библиотеки"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging() -> None:
    """Настраивает базовое логирование."""
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )
