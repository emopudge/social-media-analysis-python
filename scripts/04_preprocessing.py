import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymorphy3 import MorphAnalyzer

for pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

morph = MorphAnalyzer()
STOPWORDS = set(stopwords.words("russian")) | set(stopwords.words("english"))


def preprocess(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^а-яА-ЯёЁ\s]", " ", text)
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    lemmas = [morph.parse(t)[0].normal_form for t in tokens]
    return " ".join(lemmas)


if __name__ == "__main__":
    df = pd.read_csv("data/nerds.csv")
    df["text"] = df["text"].fillna("")
    df["clean_text"] = df["text"].apply(preprocess)
    df.to_csv("data/nerds_clean.csv", index=False)
    print(f"Processed {len(df)} posts")
