# PII Shield — ruBert NER для выделения персональных данных

Обучение BERT-модели для задачи NER (Named Entity Recognition) по выделению 30 типов PII-сущностей в русскоязычных текстах.

## Структура проекта

```
pii-shield/
├── train.py              # Скрипт обучения
├── postprocessing.py     # Постобработка предсказаний
├── requirements.txt      # Зависимости
├── data/
│   ├── train_dataset.tsv       # Основной датасет (8287 примеров)
│   └── synth_data_preproc.csv  # Синтетические данные (4856 примеров)
└── raw/                        # Исходные файлы-референсы
```

## Лучшие практики, заложенные в train.py

| Практика | Реализация |
|---|---|
| BIO-схема разметки | B-/I- префиксы для 30 сущностей + O (61 метка) |
| K-Fold CV (5 фолдов) | Усреднение вероятностей (probability blending) на инференсе |
| EMA весов | decay=0.995, применяется при eval и инференсе |
| Cosine schedule | `lr_scheduler_type="cosine"` с warmup |
| Label smoothing | 0.1 — защита от переуверенности |
| Class-weighted loss | Обратные частоты классов — борьба с дисбалансом |
| Gradient checkpointing | Экономия VRAM для max_length=512 |
| Mixed precision | fp16 / bf16 автоматически на GPU |
| Pseudo-labeling | Конфидентные предсказания на тесте добавляются в обучение |
| Постобработка | Расширение спанов до границ слов, удаление пересечений, валидация форматов |

## Установка

```bash
pip install -r requirements.txt
```

Для GPU (CUDA):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Команды обучения

### 1. Быстрая проверка (1 фолд, 2 эпохи, без синтетики)

Для отладки пайплайна — убеждается, что всё запускается без ошибок:

```bash
python train.py
```

Перед запуском отредактируйте `TrainConfig` в `train.py`:

```python
cfg = TrainConfig(
    n_folds=1,
    num_epochs=2,
    val_sample=100,
    synth_file="",              # без синтетики
    test_file="",               # без инференса
    batch_size=8,
    grad_accum=2,
)
```

### 2. Полное обучение (5 фолдов, 10 эпох, с синтетикой)

```bash
python train.py
```

Конфиг по умолчанию:

```python
cfg = TrainConfig(
    n_folds=5,
    num_epochs=10,
    batch_size=8,
    grad_accum=4,               # effective batch = 32
    learning_rate=2e-5,
    max_length=512,
    use_ema=True,
    use_pseudo_labels=False,
    synth_file="data/synth_data_preproc.csv",
    test_file="",               # не задан — только обучение
)
```

### 3. Обучение + инференс на тесте

```python
cfg = TrainConfig(
    test_file="data/private_test_dataset.csv",   # путь к тесту
    output_file="submission.csv",
)
```

### 4. Обучение с pseudo-labeling

Когда есть тестовые данные без разметки:

```python
cfg = TrainConfig(
    test_file="data/private_test_dataset.csv",
    use_pseudo_labels=True,
    uncertain_thresh=0.15,      # примеры выше — «сложные»
    confident_thresh=0.04,      # примеры ниже — «конфидентные», добавляются в обучение
)
```

### 5. Мульти-GPU (через Accelerate)

```bash
accelerate launch train.py
```

Или через `torchrun`:

```bash
torchrun --nproc_per_node=2 train.py
```

## Рекомендуемый порядок запуска

1. **Проверка пайплайна** — 1 фолд, 2 эпохи. Убедиться, что данные грузятся, токенизация корректна, метрики считаются.
2. **Подбор гиперпараметров** — 3 фолда, 5 эпох. Смотреть на val F1 по фолдам. Подобрать `learning_rate`, `threshold`, `label_smoothing`.
3. **Полное обучение** — 5 фолдов, 10 эпох, с синтетикой. Это основной запуск.
4. **Pseudo-labeling** — после основного обучения, если есть тестовые данные. Добавить конфидентные предсказания и переобучить.
5. **Ансамбль нескольких моделей** — добавить `model_name_2` (например, conversational NER), обучить отдельные фолды и объединить предсказания.

## Аппаратные требования

| Конфигурация | VRAM | Время (оценка) |
|---|---|---|
| 1 фолд, bs=8, max_len=512 | ~10 GB | ~15 мин на эпоху |
| 5 фолдов, 10 эпох | ~10 GB | ~12 часов |
| + pseudo-labeling | ~10 GB | +3-4 часа |

Если VRAM не хватает:
- Уменьшите `batch_size` до 4 и увеличьте `grad_accum` до 8 (effective batch тот же)
- Уменьшите `max_length` до 256 (если тексты короткие)
- Включите `gradient_checkpointing=True` (уже включён по умолчанию)

## Формат данных

**train_dataset.tsv** (TSV, колонки: text, target, entity):
```
text	target	entity
Мой номер +7-999-123-45-67	[(9, 26, 'Номер телефона')]	['+7-999-123-45-67']
```

**synth_data_preproc.csv** (CSV, колонки: id, text, target, entity):
```
id,text,target,entity
0,Наши запросы с API ключом bk_api_...,"[(26, 69, 'API ключи')]",['bk_api_...']
```

## Типы PII-сущностей (30 классов)

API ключи, CVV/CVC, Email, Водительское удостоверение, Временное удостоверение личности, Гражданство и названия стран, Данные об автомобиле клиента, Данные об организации/юридическом лице, Дата окончания срока действия карты, Дата регистрации по месту жительства или пребывания, Дата рождения, Имя держателя карты, Кодовые слова, Место рождения, Наименование банка, Номер банковского счета, Номер карты, Номер телефона, Одноразовые коды, ПИН код, Пароли, Паспортные данные, Полный адрес, Разрешение на работу / визу, СНИЛС клиента, Сведения об ИНН, Свидетельство о рождении, Серия и номер вида на жительство, Содержимое магнитной полосы, ФИО.
