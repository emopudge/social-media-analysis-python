# Анализ популярности постов Telegram-канала

Пайплайн анализа Telegram-канала: парсинг через API, предобработка текста, бинарная классификация популярности постов.

## Задача

По тексту поста предсказать, станет ли он популярным (число реакций выше среднего по каналу). Признаки — очищенные лемматизированные тексты и TF-IDF. Модели: LogisticRegression, LinearSVC, RandomForest.

## Данные

1000+ постов канала `@nerdsmedia`, собранных через Telethon: `id_post, date, text, views, reactions, comments`. Схема — в `data/nerds.csv`.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск

```bash
cp .env.example .env  # заполнить TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_CHANNEL
python scripts/02_parsing.py
python scripts/04_preprocessing.py
python scripts/06_classification.py
```

Метрики сохраняются в `data/metrics.csv`, модель — в `models/classifier.joblib`.

## Результаты

Метрики на тестовой выборке (таргет: `reactions > среднего` по каналу):

| Модель | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| LogisticRegression | 0.619 | 0.575 | **0.596** |
| LinearSVC | 0.606 | 0.575 | 0.590 |
| RandomForest | 0.658 | 0.524 | 0.584 |

Baseline (всегда предсказывать мажоритарный класс): F1 ≈ 0.50.

Лучшая модель по валидационной выборке — LogisticRegression (val F1 = 0.608).
Обучение на текстах без метаданных (время, просмотры, длина). Добавление этих
признаков — следующий шаг.

## Структура

```
scripts/       02_parsing.py, 04_preprocessing.py, 06_classification.py
data/          nerds.csv, nerds_clean.csv, metrics.csv
models/        tfidf.joblib, classifier.joblib
images/        engagement_dynamics.png, topics_lda.png
```

## Ограничения

- На новых аккаунтах Telegram включает FloodWait. В `02_parsing.py` реализован backoff с повтором запросов.
- Если парсинг недоступен, используйте готовый `data/nerds.csv`.

## Стек

Python 3.10+, pandas, scikit-learn, NLTK, pymorphy3, Telethon, matplotlib, seaborn.