#  A Comparative Study of Optimization Methods for Full Fine-Tuning of Qwen2.5-0.5B

## Обзор

Данный проект реализует **полное дообучение (full fine-tuning)** языковой модели **Qwen2.5-0.5B** на датасете **openwebtext-100k** с использованием трёх стратегий оптимизации:

- **AdamW** – классический адаптивный оптимизатор (базовый уровень).
- **Muon** – новый оптимизатор на основе ортогонализации матриц (обещает 2× ускорение).
- **Hybrid (Muon + AdamW)** – параметры модели разделены на две группы:  
  *внимание* (`q_proj, k_proj, v_proj`) обучается с Muon, остальные – с AdamW.
- **MeZO** *(Challenge)* – zeroth-order оптимизатор, оценивающий градиент без обратного распространения.

**Цель:** сравнить оптимизаторы по времени обучения, пиковому потреблению памяти, сходимости и итоговому качеству на наборах данных `PIQA`, `ARC-easy`, `ARC-challenge`, `WinoGrande`, `HellaSwag`.

---

## Технологии проекта

Проект реализован на **Python 3.10** и использует современный стек для обучения LLM.

### Управление зависимостями

Проект поддерживает два способа управления зависимостями:

- **Poetry** – для воспроизводимого окружения и удобной публикации. Файл `pyproject.toml` содержит все зависимости с фиксированными версиями.
- **requirements.txt** – предоставлен для обратной совместимости и простоты установки в средах без Poetry.

Установка через Poetry:
```bash
poetry install
poetry shell
```

Установка через pip:
```bash
pip install -r requirements.txt
```


## Запуск экспериментов

### AdamW
```bash
scripts/run_adamw.sh
```
run_adamw.sh owerview for example:
```bash

python src/train.py \
  --model_name Qwen/Qwen2.5-0.5B \
  --optimizer adamw \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 3e-4 \
  --seq_length 512 \
  --num_train_epochs 1 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.01 \
  --weight_decay 0.01 \
  --logging_steps 10 \
  --push_to_hub True \
  --report_to clearml \
  --seed 42 \
  --output_dir ./Qwen2.5-0.5B_muon
```
### Muon
```bash
scripts/run_muon.sh
```

### HybridMuon
```bash
scripts/run_muon_hybrid.sh
```

### MeZO
```bash
scripts/run_mezo.sh
```

## 📓 Запуск проекта в Google Colab

В репозитории есть готовый ноутбук `notebooks/run_experiments.ipynb`, который автоматизирует весь процесс: установку зависимостей, загрузку модели и датасета, запуск экспериментов с AdamW, Muon, Hybrid и MeZO, логирование в TensorBoard/ClearML, а также визуализацию результатов.

### 🚀 Пошаговая инструкция

1. **Откройте Google Colab**  
   Перейдите на [colab.research.google.com](https://colab.research.google.com).

2. **Загрузите ноутбук из репозитория**  
   - Он находится здесь notebooks/GoogleColabRun.ipynb

3. **Включите GPU-ускоритель**  
   В меню `Runtime` → `Change runtime type` → выберите `T4 GPU` (или `V100`/`A100` для Colab Pro).

4. **Запустите все ячейки**  
   Нажмите `Runtime` → `Run all`. Ноутбук автоматически:
   - Смонтирует Google Drive (опционально, для сохранения результатов).
   - Установит зависимости из `requirements.txt`.
   - Загрузит модель `Qwen2.5-0.5B` и датасет `openwebtext-100k`.






## Методы

### 1. AdamW
- Реализация `torch.optim.AdamW`

### 2. Muon
- Исходный код: [Moonlight](https://github.com/MoonshotAI/Moonlight)
- Особенности: Newton-Schulz итерации для ортогонализации, адаптивное масштабирование

### 3. Гибридный Muon
- Группа 1 (Muon): параметры `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Группа 2 (AdamW): все остальные параметры
- Используется единый `Optimizer` с двумя `param_groups`

### 4. MeZO (Zeroth-Order)
- Реализация из [Princeton-NLP/MeZO](https://github.com/princeton-nlp/MeZO)
- Данная реализация доработана для работы с `transformers>=4.5`
- Сходимость только на основе двух forward-проходов на шаг

---

## ⚙️ Экспериментальная установка

| Параметр            | Значение                     |
|---------------------|------------------------------|
| Model              | `Qwen/Qwen2.5-0.5B`          |
| Dataset             | `Elriggs/openwebtext-100k (only 10k samples)`   |
| Evaluation              | `lm-evaluation-harness`      |
| Batch size   | 2 (gradient accumulation=8)  |
| Epochs               | 1                            |
| Sequence length| 512                          |
| Precision            | `float32`  |
| GPU                 | NVIDIA T4 (15GB) / Colab (free) |

---

## Results

### Время и память

| Оптимизатор   | Время (мин) | Пик памяти (GB) |
|---------------|-------------|-----------------|
| AdamW         |    |        |
| Muon          |    |        |
| Hybrid        |    |        |
| MeZO          |    |        |

### Evaluation (accuracy %)

| Модель / Оптимизатор | PIQA | ARC-e | ARC-c | WinoGrande | HellaSwag |
|----------------------|------|-------|-------|------------|-----------|
| Без fine-tuning      |      |  |    |       |                        |
| + AdamW              |  |  |   |        |     |
| + Muon               |  |  |   |      |      |
| + Hybrid             |  |  | |  |  |
| + MeZO               | |  |  |       |     |



## 

### Требования

- Python 3.10+
- CUDA 11.8+ (для GPU)
- Установка зависимостей:

```bash
pip install -r requirements.txt