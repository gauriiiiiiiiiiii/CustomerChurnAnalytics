# Customer Churn Analytics + Business Dashboard

End-to-end churn prediction with advanced feature engineering, model comparison, SHAP explainability, FastAPI inference, and Streamlit dashboard.

Quick start:
```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.data_prep
python -m src.train
python -m src.explain
.\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000
.\.venv\Scripts\streamlit.exe run src/dashboard.py
```

