from pathlib import Path
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import load_data

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "complaints.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

df = load_data(str(DATA))

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["category"],
    test_size=0.20, random_state=42, stratify=df["category"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True)),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

joblib.dump(model, MODELS / "category_model.joblib")

metrics = {
    "model": "TF-IDF + Logistic Regression",
    "accuracy": round(float(accuracy), 4),
    "classification_report": classification_report(
        y_test, pred, output_dict=True
    )
}

with open(REPORTS / "model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, default=float)

print(f"Category model accuracy: {accuracy:.2%}")
print(f"Saved: {MODELS / 'category_model.joblib'}")
