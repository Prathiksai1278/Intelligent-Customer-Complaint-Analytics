from pathlib import Path
import sqlite3
from preprocessing import load_data

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "complaints.db"
DATA = ROOT / "data" / "complaints.csv"

df = load_data(str(DATA))
df = df.drop(columns=["clean_text"])

conn = sqlite3.connect(DB)
df.to_sql("complaints", conn, if_exists="replace", index=False)

conn.execute("""
CREATE INDEX IF NOT EXISTS idx_category ON complaints(category)
""")
conn.execute("""
CREATE INDEX IF NOT EXISTS idx_status ON complaints(status)
""")
conn.commit()
conn.close()

print(f"Database created: {DB}")
