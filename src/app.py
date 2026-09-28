from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import load_data
from predictor import predict

DATA = ROOT / "data" / "complaints.csv"

st.set_page_config(
    page_title="Complaint Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Intelligent Customer Complaint Analytics")
st.caption("Enterprise customer-service analytics and machine-learning dashboard")

df = load_data(str(DATA))

# Sidebar
st.sidebar.header("Filters")
categories = st.sidebar.multiselect(
    "Category",
    sorted(df["category"].unique()),
    default=sorted(df["category"].unique())
)
statuses = st.sidebar.multiselect(
    "Status",
    sorted(df["status"].unique()),
    default=sorted(df["status"].unique())
)
priorities = st.sidebar.multiselect(
    "Priority",
    sorted(df["priority"].unique()),
    default=sorted(df["priority"].unique())
)

filtered = df[
    df["category"].isin(categories) &
    df["status"].isin(statuses) &
    df["priority"].isin(priorities)
]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Complaints", f"{len(filtered):,}")
c2.metric("High Priority", f"{(filtered['priority']=='High').sum():,}")
c3.metric("Negative Sentiment", f"{(filtered['sentiment']=='Negative').sum():,}")

resolved = filtered.loc[
    filtered["status"] == "Resolved", "resolution_days"
].dropna()
avg = resolved.mean() if len(resolved) else 0
c4.metric("Avg Resolution", f"{avg:.1f} days")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Complaints by Category")
    st.bar_chart(filtered["category"].value_counts())

with right:
    st.subheader("Sentiment Distribution")
    st.bar_chart(filtered["sentiment"].value_counts())

left, right = st.columns(2)

with left:
    st.subheader("Priority Distribution")
    st.bar_chart(filtered["priority"].value_counts())

with right:
    st.subheader("Complaint Channels")
    st.bar_chart(filtered["channel"].value_counts())

st.subheader("Monthly Complaint Trend")
monthly = (
    filtered.set_index("date")
    .resample("ME")
    .size()
    .rename("complaints")
)
st.line_chart(monthly)

st.subheader("Recent Complaints")
display_cols = [
    "complaint_id", "date", "category", "priority",
    "sentiment", "status", "channel", "complaint_text"
]
st.dataframe(
    filtered.sort_values("date", ascending=False)[display_cols].head(20),
    use_container_width=True,
    hide_index=True
)

st.divider()
st.subheader("🔎 Analyze a New Complaint")

text = st.text_area(
    "Enter a customer complaint",
    placeholder="Example: I was charged twice and nobody has resolved the issue."
)

if st.button("Analyze Complaint", type="primary"):
    if not text.strip():
        st.warning("Please enter a complaint.")
    else:
        try:
            result = predict(text)
            a, b, c = st.columns(3)
            a.metric("Predicted Category", result["category"])
            b.metric("Sentiment", result["sentiment"])
            c.metric("Priority", result["priority"])
        except FileNotFoundError as exc:
            st.error(str(exc))

st.divider()
st.caption("Synthetic dataset • Built for academic and portfolio demonstration")
