from typing import List
import pandas as pd
from fastapi import FastAPI

from src.data_prep import clean_data, load_raw_data
from src.features import add_features
from src.insights import generate_insights
from src.schemas import PredictRequest, PredictResponse, PredictResponseItem
from src.train import train_best_model

app = FastAPI(title="Customer Churn Analytics API")


@app.get("/")
def home():
    return {"message": "Customer Churn API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    feature_df = add_features(clean_df)
    model = train_best_model(feature_df, save_artifacts=False)

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
