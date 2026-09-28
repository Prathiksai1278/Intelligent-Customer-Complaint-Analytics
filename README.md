# Intelligent Customer Complaint Analytics Platform

A Python-based analytics and machine-learning platform for analyzing customer complaints, identifying recurring issues, classifying complaints, estimating priority, analyzing sentiment, and visualizing customer-service performance.

## Features

- Complaint dataset ingestion using CSV
- Text preprocessing
- Automated complaint category classification
- Sentiment analysis
- Priority prediction
- Exploratory data analysis
- Complaint trend analysis
- Resolution-time analysis
- Interactive Streamlit dashboard
- SQLite database storage
- Model persistence with Joblib
- Automated CSV report generation

## Architecture

CSV Dataset
→ Data Preprocessing
→ NLP Feature Extraction
→ ML Classification
→ Sentiment & Priority Analysis
→ SQLite Storage
→ Analytics Dashboard

## Technology

Python, Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn, SQLite, Streamlit, Joblib

## Installation

### 1. Create virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train models

```bash
python src/train_models.py
```

### 4. Build the database and generate reports

```bash
python src/build_database.py
python src/generate_report.py
```

### 5. Start dashboard

```bash
streamlit run src/app.py
```

Open the URL displayed by Streamlit, normally:

http://localhost:8501

## Project Structure

```text
Intelligent_Customer_Complaint_Analytics/
│
├── data/
│   └── complaints.csv
├── models/
├── reports/
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── predictor.py
│   ├── build_database.py
│   ├── generate_report.py
│   └── app.py
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
└── README.md
```

## ML Approach

The complaint category model uses:

- TF-IDF vectorization
- Logistic Regression
- Train/test split
- Accuracy and classification report

The project intentionally uses an interpretable classical ML pipeline rather than a large language model so that the complete workflow can be understood and explained during interviews.

## Resume Project Description

**Intelligent Customer Complaint Analytics Platform**

- Developed a Python-based analytics platform to analyze and classify customer complaints using natural language processing and machine learning.
- Implemented text preprocessing, TF-IDF feature extraction and supervised learning for automated complaint categorization.
- Performed exploratory data analysis to identify recurring customer issues, complaint trends and resolution patterns.
- Built an interactive dashboard to visualize complaint categories, sentiment, priority levels and service performance.
- Generated automated analytical reports to support customer-service teams in identifying high-priority and recurring issues.

## Important

The included dataset is synthetic and is intended for demonstration and learning purposes.
