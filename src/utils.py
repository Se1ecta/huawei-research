"""Утилиты используемые в проекте."""

import random

import torch


def set_seed(seed: int) -> None:
    """Установка random seed в различные библиотеки"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
