from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "category_model.joblib"

NEGATIVE_WORDS = {
    "bad", "terrible", "frustrating", "unacceptable", "disappointed",
    "angry", "worst", "failed", "failure", "broken", "late", "wrong",
    "cannot", "unable", "issue", "problem"
}

HIGH_WORDS = {
    "urgent", "critical", "twice", "charged", "locked", "crashing",
    "unacceptable", "fraud", "security", "cannot access"
}

def sentiment(text):
    words = set(str(text).lower().split())
    negative = len(words & NEGATIVE_WORDS)
    positive = sum(w in words for w in {"thanks", "thank", "great", "good", "appreciate"})
    if negative > positive:
        return "Negative"
    if positive > negative:
        return "Positive"
    return "Neutral"

def priority(text):
    low = str(text).lower()
    if any(word in low for word in HIGH_WORDS):
        return "High"
    if any(word in low for word in {"delay", "slow", "error", "problem", "issue"}):
        return "Medium"
    return "Low"

def predict(text):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run: python src/train_models.py"
        )
    model = joblib.load(MODEL_PATH)
    category = model.predict([text])[0]
    return {
        "category": category,
        "sentiment": sentiment(text),
        "priority": priority(text)
    }

if __name__ == "__main__":
    text = input("Enter complaint: ")
    print(predict(text))
