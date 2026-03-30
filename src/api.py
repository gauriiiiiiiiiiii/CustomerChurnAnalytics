from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI

from src.config import MODEL_PATH
from src.features import add_features
from src.insights import generate_insights
from src.schemas import PredictRequest, PredictResponse, PredictResponseItem

app = FastAPI(title="Customer Churn Analytics API")

model = None


@app.on_event("startup")
def load_model() -> None:
    global model
    model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    df = pd.DataFrame([r.dict() for r in request.records])
    df = add_features(df)

    proba = model.predict_proba(df)[:, 1]
    preds = (proba >= 0.5).astype(int)

    results: List[PredictResponseItem] = []
    for i, row in df.iterrows():
        insights = generate_insights(row.to_dict(), float(proba[i]))
        results.append(
            PredictResponseItem(
                customerID=str(row.get("customerID", i)),
                churn_probability=float(proba[i]),
                churn_label="Yes" if preds[i] == 1 else "No",
                insights=insights["insights"],
            )
        )

    return PredictResponse(predictions=results)
