# Customer Churn Analytics + Business Dashboard

Minimal, production-ready churn prediction system:
- FastAPI backend that loads a pre-trained model and serves predictions.
- Streamlit dashboard that calls the API (no local CSVs or training at runtime).

Quick start (local):
```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# API (requires models/best_model.joblib)
\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000

# Dashboard (calls the deployed API URL in src/dashboard.py)
\.venv\Scripts\streamlit.exe run src/dashboard.py
```

