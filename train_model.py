from pathlib import Path
import pickle
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "spam.csv"
MODEL_PATH = BASE_DIR / "spam_model.pkl"

# The notebook uses these NLTK resources.
# If they are already installed, these calls do nothing.
for resource, package in [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(package)

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()


def transform(txt):
    txt = txt.lower()
    txt = nltk.word_tokenize(txt)
    txt = [word for word in txt if word.isalnum()]
    txt = [word for word in txt if word not in stop_words]
    txt = [ps.stem(word) for word in txt]
    return " ".join(txt)


if not DATA_PATH.exists():
    raise FileNotFoundError(
        "spam.csv was not found. Put the same spam.csv used in the notebook "
        "in this folder and run this script again."
    )

df = pd.read_csv(DATA_PATH, encoding="latin-1")
df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
df = df.rename(columns={"v1": "target", "v2": "text"})

# Same label encoding used in the notebook.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["target"] = le.fit_transform(df["target"])

# Same duplicate removal used in the notebook.
df = df.drop_duplicates(keep="first")

# Same NLP transformation used in the notebook.
df["transformed"] = df["text"].apply(transform)

# Final model from the notebook:
# TF-IDF(max_features=3000) + MultinomialNB
X = df["transformed"]
y = df["target"]

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=3000)),
    ("mnb", MultinomialNB()),
])

model.fit(X, y)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to: {MODEL_PATH}")
print("Pipeline: preprocessing -> TF-IDF(max_features=3000) -> MultinomialNB")
