# Сравнение оптимзаторов для полного дообучения Qwen2.5-0.5B

## 📌 Обзор

Данный проект реализует **полное дообучение (full fine-tuning)** языковой модели **Qwen2.5-0.5B** на датасете **openwebtext-100k** (10k семплов) с использованием четырёх стратегий оптимизации:

- **AdamW** – классический адаптивный оптимизатор (базовый уровень).
- **Muon** – новый оптимизатор на основе ортогонализации матриц (обещает 2× ускорение сходимости).
- **Muon for attnetion matrices ** – параметры проекций внимания (`q_proj, k_proj, v_proj, o_proj`) обучаются с Muon, остальные – с AdamW.
- **MeZO** *(Challenge)* – zeroth-order оптимизатор, оценивающий градиент без обратного распространения (только два forward-прохода на шаг).

**Цель:** сравнить оптимизаторы по времени обучения, пиковому потреблению памяти, loss и итоговому качеству на стандартных бенчмарках: `PIQA`, `ARC-easy`, `ARC-challenge`, `WinoGrande`, `HellaSwag`.

---

## 📓 Запуск в Google Colab / Kaggle (без локальной установки)

В репозитории подготовлен ноутбук **`notebooks/ExperimentRun.ipynb`**, который автоматически:
- Определяет среду (Colab/Kaggle/local)
- Монтирует Google Drive (при необходимости)
- Клонирует репозиторий и устанавливает зависимости
- Настраивает ClearML через Colab Secrets / Kaggle Secrets
- Запускает **серию экспериментов** (AdamW, Muon, Hybrid, MeZO) с заданными конфигурациями
- После обучения запускает **оценку моделей** через `lm-evaluation-harness` (PIQA, ARC, WinoGrande, HellaSwag)
- Строит сводную таблицу и график сравнения

## 🚀 Быстрый старт

### Требования
- Python 3.10+
- GPU с 12+ GB VRAM (рекомендуется T4/V100/A10)
- Установленные CUDA и PyTorch 2.0+

### Установка

#### Способ 1: Poetry (рекомендуется для воспроизводимости)
```bash
git clone https://github.com/Se1ecta/huawei-research.git
cd huawei-research
poetry install
poetry shell
```

#### Способ 2: pip + requirements.txt
```bash
git clone https://github.com/Se1ecta/huawei-research.git
cd huawei-research
pip install -r requirements.txt
```

Добавим в README раздел про настройку переменных окружения через `.env` файл. В текущей версии проекта, вероятно, присутствует файл `.env.example` с шаблоном для `CLEARML_API_ACCESS_KEY`, `CLEARML_API_SECRET_KEY`, `HF_TOKEN` и т.д. Пользователь должен скопировать его и заполнить.

Ниже — **обновлённый фрагмент README** (вставляется после раздела установки зависимостей, перед запуском экспериментов). Также добавим упоминание, что в Colab/Kaggle используются встроенные секреты.

---

## 🔐 Настройка переменных окружения

Для локального запуска экспериментов (через скрипты `scripts/*.sh`) рекомендуется использовать файл `.env` для хранения sensitive данных (ключи ClearML, Hugging Face токен).

1. Скопируйте шаблон:
   ```bash
   cp .env.example .env
   ```

2. Отредактируйте `.env`:


3. Загрузите переменные в окружение перед запуском:
   ```bash
   source .env
   # или используйте python-dotenv внутри скриптов
   ```

Полный обновлённый README с этим блоком можно вставить в файл. Если нужно, я могу прислать весь README целиком с уже интегрированным разделом.

#### Настройка ClearML (опционально, для отслеживания экспериментов)
```bash
clearml-init
```
Или установите переменные окружения:
```bash
export CLEARML_API_ACCESS_KEY="..."
export CLEARML_API_SECRET_KEY="..."
```

#### Авторизация Hugging Face Hub (для пуша моделей)
```bash
huggingface-cli login
```

---

## 🏃 Запуск экспериментов

Все скрипты запуска находятся в папке `scripts/`. Примеры:

### AdamW (базовый)
```bash
bash scripts/run_adamw.sh
```
Содержимое `run_adamw.sh`:
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
  --output_dir ./Qwen2.5-0.5B_adamw
```

### Muon
```bash
bash scripts/run_muon.sh
```

### Hybrid Muon (Muon на attention-слоях, AdamW на остальных)
```bash
bash scripts/run_muon_hybrid.sh
```

### MeZO (Zeroth‑order)
```bash
bash scripts/run_mezo.sh
```

> **Примечание:** Для MeZO рекомендуется увеличить `zo_eps` (например, `--zo_eps 1e-3`) и возможно снизить `learning_rate`.

---



### Как использовать ноутбук:

1. Откройте [Google Colab](https://colab.research.google.com/)  
2. Загрузите ноутбук: `File → Upload notebook` → выберите `notebooks/run_experiments.ipynb`  
3. Включите GPU: `Runtime → Change runtime type → T4 GPU`  
4. (Рекомендуется) Добавьте секреты:
   - `CLEARML_API_ACCESS_KEY` и `CLEARML_API_SECRET_KEY` (для трекинга)
   - `HF_TOKEN` (опционально, для пуша модели на Hub)  
5. Запустите все ячейки: `Runtime → Run all`

Ноутбук сам выполнит клонирование, установку, обучение и оценку. Результаты появятся в папке `outputs/` и в веб-интерфейсе ClearML.

---

## ⚙️ Параметры экспериментов

| Параметр            | Значение                              |
|---------------------|---------------------------------------|
| Модель              | `Qwen/Qwen2.5-0.5B`                   |
| Датасет             | `Elriggs/openwebtext-100k` (10k samples) |
| Оценка              | `lm-evaluation-harness` (git-версия)  |
| Batch size          | 2 (gradient accumulation = 8) → эфф. batch 16 |
| Эпохи               | 1                                     |
| Длина последовательности | 512                              |
| Точность            | `float32`  |
| GPU                 | NVIDIA T4 (15GB) / V100 / A10         |
| LR scheduler        | cosine с warmup ratio 0.01            |
| Weight decay        | 0.01 |

---

## 🧪 Методы

### 1. AdamW
- Реализация `torch.optim.AdamW`

### 2. Muon
- Исходный код: [Moonlight](https://github.com/MoonshotAI/Moonlight)
- Особенности: Newton-Schulz итерации для ортогонализации градиента, адаптивное масштабирование

### 3. Muon for attention matrices
- **Группа Muon**: параметры `q_proj, k_proj, v_proj, o_proj` (все веса внимания)
- **Группа AdamW**: все остальные параметры (embedding, LM head и тд. )

### 4. MeZO (Zeroth-Order)
- Реализация на основе [Princeton-NLP/MeZO](https://github.com/princeton-nlp/MeZO), адаптированная для `transformers>=4.5`
- Оценка градиента: `∇L ≈ (L(θ+εz) - L(θ-εz)) / (2ε) * z`
- Требует два forward-прохода на шаг, **без backward** → экономия памяти на градиентах.

---

## 📊 Результаты

> **Примечание:** таблицы будут заполнены после прогона экспериментов. Ниже приведены ожидаемые диапазоны на основе литературы.

### Время и память (1 эпоха, 10k семплов, batch=16 effective)

| Оптимизатор   | Время (мин) | Пик памяти (GB) |
|---------------|-------------|-----------------|
| AdamW         | TBD         | TBD             |
| Muon          | TBD         | TBD             |
| Hybrid        | TBD         | TBD             |
| MeZO          | TBD         | TBD (ожидается ~30% меньше AdamW) |

### Accuracy (%) на задачах reasoning

| Модель / Оптимизатор | PIQA | ARC-e | ARC-c | WinoGrande | HellaSwag |
|----------------------|------|-------|-------|------------|-----------|
| Без fine-tuning      | TBD  | TBD   | TBD   | TBD        | TBD       |
| + AdamW              | TBD  | TBD   | TBD   | TBD        | TBD       |
| + Muon               | TBD  | TBD   | TBD   | TBD        | TBD       |
| + Hybrid             | TBD  | TBD   | TBD   | TBD        | TBD       |
| + MeZO               | TBD  | TBD   | TBD   | TBD        | TBD       |

*Оценка проведена с помощью `lm_eval --model hf` (git-версия).*

---

## 🛠 Структура проекта

```
huawei-research/
├── notebooks/
│   └── ExperimentRun.ipynb   # автоматизированный ноутбук для Colab/Kaggle
├── scripts/
│   ├── run_adamw.sh
│   ├── run_muon.sh
│   ├── run_muon_hybrid.sh
│   └── run_mezo.sh
├── src/
│   ├── train.py                # основной скрипт обучения
│   ├── optimizers/             # реализации Muon
│   ├── mezo/                   # реализации Mezo


├── requirements.txt
├── pyproject.toml              # для Poetry
└── README.md
```