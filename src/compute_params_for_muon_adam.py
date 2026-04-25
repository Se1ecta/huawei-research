"""
Скрипт для отображения распределения параметров модели между Muon и Adam.
Режимы:
  - "muon": в Muon идут все 2D-параметры, кроме embed_tokens, lm_head, norm.
  - "hybrid_muon": в Muon идут только 2D-параметры из attention слоёв (содержат "attn").
Остальные параметры попадают в Adam.
"""  # noqa: D205

from transformers import Qwen2Config, Qwen2ForCausalLM

MODE = "hybrid_muon"  # или "hybrid_muon"


def build_model():
    """Создаёт модель Qwen2 с параметрами из условия."""
    config = Qwen2Config(
        attention_dropout=0.0,
        bos_token_id=151643,
        eos_token_id=151643,
        hidden_act="silu",
        hidden_size=1024,  # для демонстрации взят небольшой размер
        initializer_range=0.02,
        intermediate_size=4864,
        max_position_embeddings=513,
        max_window_layers=12,
        model_type="qwen2",
        num_attention_heads=16,
        num_hidden_layers=12,
        num_key_value_heads=16,
        rms_norm_eps=1e-06,
        rope_theta=1000000.0,
        sliding_window=1024,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        use_cache=True,
        use_mrope=False,
        use_sliding_window=False,
        vocab_size=151936,
    )
    model = Qwen2ForCausalLM(config)
    return model


def split_parameters(model, mode):
    """Возвращает два списка: (muon_params, adam_params) как кортежи (имя, тензор)."""
    muon_params = []
    adam_params = []

    for name, param in model.named_parameters():
        if mode == "muon":
            if param.ndim == 2 and "embed_tokens" not in name and "lm_head" not in name and "norm" not in name:
                muon_params.append((name, param))
            else:
                adam_params.append((name, param))

        elif mode == "hybrid_muon":
            if "attn" in name and param.ndim == 2:
                muon_params.append((name, param))
            else:
                adam_params.append((name, param))

        else:
            raise ValueError(f"Unknown mode: {mode}")

    return muon_params, adam_params


def print_parameter_group(group, group_name):
    """Выводит информацию о группе параметров."""
    print(f"\n=== {group_name} ({len(group)} параметров) ===")
    total_params = 0
    for name, param in group:
        num_params = param.numel()
        total_params += num_params
        print(f"  {name:<70} shape={tuple(param.shape)} params={num_params:,}")
    print(f"  Итого параметров в группе: {total_params:,}\n")


def main():
    print(f"Загрузка модели Qwen2 (режим: {MODE})...")
    model = build_model()
    print("Модель загружена.\n")

    muon_group, adam_group = split_parameters(model, MODE)

    print_parameter_group(muon_group, "Параметры для Muon")
    print_parameter_group(adam_group, "Параметры для Adam (или других оптимизаторов)")

    total_muon = sum(p.numel() for _, p in muon_group)
    total_adam = sum(p.numel() for _, p in adam_group)
    total_all = total_muon + total_adam
    print(f"Всего параметров в модели: {total_all:,}")
    print(f"Muon: {total_muon:,} ({total_muon / total_all * 100:.2f}%)")
    print(f"Adam: {total_adam:,} ({total_adam / total_all * 100:.2f}%)")


if __name__ == "__main__":
    main()
