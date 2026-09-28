from pathlib import Path
import pandas as pd
from preprocessing import load_data

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "complaints.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

df = load_data(str(DATA))

category = df["category"].value_counts().rename_axis("category").reset_index(name="complaints")
sentiment = df["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="complaints")
priority = df["priority"].value_counts().rename_axis("priority").reset_index(name="complaints")
status = df["status"].value_counts().rename_axis("status").reset_index(name="complaints")

category.to_csv(REPORTS / "category_summary.csv", index=False)
sentiment.to_csv(REPORTS / "sentiment_summary.csv", index=False)
priority.to_csv(REPORTS / "priority_summary.csv", index=False)
status.to_csv(REPORTS / "status_summary.csv", index=False)

resolved = df.loc[df["status"] == "Resolved", "resolution_days"].dropna()
summary = pd.DataFrame([{
    "total_complaints": len(df),
    "resolved": int((df["status"] == "Resolved").sum()),
    "in_progress": int((df["status"] == "In Progress").sum()),
    "open": int((df["status"] == "Open").sum()),
    "high_priority": int((df["priority"] == "High").sum()),
    "negative_sentiment": int((df["sentiment"] == "Negative").sum()),
    "average_resolution_days": round(float(resolved.mean()), 2) if len(resolved) else 0
}])
summary.to_csv(REPORTS / "overall_summary.csv", index=False)

print("Reports generated in:", REPORTS)
