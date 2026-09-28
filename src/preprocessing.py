import re
import pandas as pd

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "complaint_id", "date", "category", "complaint_text",
        "priority", "status", "resolution_days", "sentiment", "channel"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df["clean_text"] = df["complaint_text"].map(clean_text)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

if __name__ == "__main__":
    df = load_data("data/complaints.csv")
    print(df.head())
