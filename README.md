# Customer Churn Analytics

Production-ready churn prediction system for telecom customers.

- **FastAPI** backend — loads a trained ML model and serves predictions
- **Streamlit** dashboard — professional UI that calls the API and displays results

Dataset: IBM Telco Customer Churn (Kaggle) — 7,043 rows  
Model: Logistic Regression — ROC-AUC 0.8465

---

## Project Structure

```
CustomerChurnAnalytics/
├── data/raw/telco_churn.csv          IBM Telco dataset (7,043 rows)
├── models/best_model.joblib          Trained sklearn Pipeline
├── src/
│   ├── preprocess.py                 Data cleaning + column definitions
│   ├── features.py                   Feature engineering (11 features)
│   ├── train.py                      End-to-end training pipeline
│   ├── api.py                        FastAPI server
│   ├── dashboard.py                  Streamlit UI
│   ├── schemas.py                    Pydantic request/response models
│   ├── insights.py                   Rule-based business recommendations
│   └── config.py                     Model path config
├── Project.txt                       Full project documentation
├── requirements.txt
└── runtime.txt
```

---

## Quick Start

**1. Install dependencies**
```powershell
pip install -r requirements.txt
```

**2. Train the model**
```powershell
python src/train.py
```

**3. Start the API (Terminal 1)**
```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

**4. Start the dashboard (Terminal 2)**
```powershell
streamlit run src/dashboard.py
```

Dashboard opens at: `http://localhost:8501`  
API docs at: `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Liveness check |
| GET | `/health` | Model load status |
| POST | `/predict` | Batch churn prediction |

---

## Model Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8465 |
| Recall | 0.537 |
| Precision | 0.681 |
| F1 | 0.601 |

See `Project.txt` for complete documentation.
