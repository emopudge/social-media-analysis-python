# Анализ популярности постов Telegram-канала

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

End-to-end пайплайн анализа социальных медиа: парсинг Telegram-канала через API,
предобработка текстов, тематическое моделирование и предсказание популярности
постов (бинарная классификация) на основе эмбеддингов Word2Vec.

## О проекте

**Задача:** по данным Telegram-канала предсказать, станет ли пост популярным
(число реакций выше среднего по каналу).

**Зачем:** инструмент для контент-аналитики — понимать, какие темы и форматы
постов дают вовлечённость, и прогнозировать её до публикации.

**Данные:** 1000+ постов Telegram-канала, собранных через Telethon API:
id, дата, текст, просмотры, реакции, комментарии.

## Пайплайн

| Шаг | Ноутбук | Что делает |
|-----|---------|------------|
| 1 | `01_python_basics.ipynb` | Справочник по базовому Python для анализа данных |
| 2 | `02_telegram_parsing.ipynb` | Парсинг постов через Telethon → DataFrame → CSV |
| 3 | `03_visualization_lda.ipynb` | EDA, визуализации, тематическое моделирование LDA + pyLDAvis |
| 4 | `04_preprocessing.ipynb` | Очистка текста, лемматизация, создание таргета, train/val/test |
| 5 | `05_word2vec.ipynb` | Обучение Word2Vec (300d) и усреднение векторов по документам |
| 6 | `06_classification.ipynb` | Сравнение классификаторов, подбор гиперпараметров, метрики |

## Результаты

Метрики на тестовой выборке (таргет: `reactions > среднего`):

| Модель | Precision | Recall | F1 |
|--------|-----------|--------|----|
| Logistic Regression | … | … | … |
| LinearSVC | … | … | … |
| Random Forest | … | … | **0.73** |

**Лучшая модель:** Random Forest, F1 = 0.73 на тесте
(подбор гиперпараметров по валидационной выборке, сохранение через `joblib`).

## Визуализации

| Вовлечённость во времени | Темы LDA |
|--------------------------|----------|
| ![](images/engagement_dynamics.png) | ![](images/topics_lda.png) |

## Стек

- **Данные:** Telethon, pandas, NumPy
- **NLP:** NLTK, pymorphy3, gensim (Word2Vec), scikit-learn (LDA, TF-IDF)
- **ML:** scikit-learn — LogisticRegression, LinearSVC, RandomForest, DecisionTree
- **Визуализация:** Matplotlib, Seaborn, Plotly, pyLDAvis, WordCloud

## Запуск

```bash
git clone https://github.com/<username>/telegram-popularity-analysis.git
cd telegram-popularity-analysis
pip install -r requirements.txt
```

Ноутбуки открываются в Jupyter/Colab. Для парсинга нужны свои `api_id` и
`api_hash` с [my.telegram.org](https://my.telegram.org) — положите их в
переменные окружения, **не коммитьте в репозиторий**.

## Структура

```
notebooks/   # 6 ноутбуков в порядке пайплайна
data/        # описание данных (CSV не хранятся в репо)
models/      # сохранённые модели
images/      # графики для README
```

## Автор

**Эмилия** — junior data analyst
- GitHub: [@emopudge](https://github.com/username)
- Telegram: @smellless_kozulya
