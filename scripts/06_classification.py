from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import LinearSVC


def load_splits():
    df = pd.read_csv("data/nerds_clean.csv")
    df["clean_text"] = df["clean_text"].fillna("")
    df["popular"] = (df["reactions"] > df["reactions"].mean()).astype(int)

    train, test = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["popular"]
    )
    train, val = train_test_split(
        train, test_size=0.15, random_state=42, stratify=train["popular"]
    )
    return train, val, test


def evaluate(model, X, y, name):
    pred = model.predict(X)
    return {
        "model": name,
        "precision": round(precision_score(y, pred, zero_division=0), 3),
        "recall": round(recall_score(y, pred, zero_division=0), 3),
        "f1": round(f1_score(y, pred, zero_division=0), 3),
    }


if __name__ == "__main__":
    train, val, test = load_splits()

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train["clean_text"])
    X_val = vectorizer.transform(val["clean_text"])
    X_test = vectorizer.transform(test["clean_text"])
    y_train, y_val, y_test = train["popular"], val["popular"], test["popular"]

    models = {
        "LogisticRegression": (LogisticRegression(max_iter=1000), {"C": [0.1, 1, 10]}),
        "LinearSVC": (LinearSVC(dual="auto", max_iter=5000), {"C": [0.1, 1, 10]}),
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100, 300], "max_depth": [5, 10, None]},
        ),
    }

    results, best_f1, best_model, best_name = [], 0, None, None

    for name, (model, params) in models.items():
        gs = GridSearchCV(model, params, cv=3, scoring="f1", n_jobs=-1)
        gs.fit(X_train, y_train)
        val_f1 = f1_score(y_val, gs.predict(X_val), zero_division=0)
        results.append(evaluate(gs.best_estimator_, X_test, y_test, name))
        print(f"{name}: val F1={val_f1:.3f}, params={gs.best_params_}")
        if val_f1 > best_f1:
            best_f1, best_model, best_name = val_f1, gs.best_estimator_, name

    Path("models").mkdir(exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf.joblib")
    joblib.dump(best_model, "models/classifier.joblib")
    print(f"\nBest on validation: {best_name}")

    metrics = pd.DataFrame(results)
    print(metrics.to_string(index=False))
    metrics.to_csv("data/metrics.csv", index=False)
