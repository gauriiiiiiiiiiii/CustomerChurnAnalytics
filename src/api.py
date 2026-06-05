# src/api.py
# FastAPI inference server -- the core backend of the churn prediction system.
#
# Loads the trained sklearn Pipeline ONCE at startup (not per request).
# The .joblib contains both the ColumnTransformer (encoder + scaler)
# and the trained classifier -- everything needed for inference in one object.
#
# Endpoints:
#   GET  /         -> liveness ping
#   GET  /health   -> model load status
#   POST /predict  -> batch churn prediction with business insights
#
# Request -> Response flow:
#   JSON -> Pydantic validation -> DataFrame
#   -> fill missing cols -> add_features()
#   -> fill again -> align cols -> predict_proba()
#   -> threshold 0.5 -> generate_insights() -> JSON

from typing import List, Tuple

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config import MODEL_PATH
from src.features import add_features
from src.insights import generate_insights
from src.schemas import PredictRequest, PredictResponse, PredictResponseItem

app = FastAPI(title="Customer Churn Analytics API")

# Load model once at startup -- avoids disk I/O on every request.
# If models/best_model.joblib is missing, model = None and /predict returns HTTP 503.
try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None


def _extract_columns() -> Tuple[List[str], List[str]]:
    """
    Read the exact column lists from the model's ColumnTransformer.
    Transformer names 'cat' and 'num' were set in preprocess.build_preprocessor().
    Returns (cat_cols, num_cols) in the same order the preprocessor was fitted on.
    This guarantees column order at inference matches training -- silent mispredictions
    would occur if columns were passed in a different order.
    """
    preprocessor = model.named_steps.get("preprocessor") if model else None
    if preprocessor is None:
        return [], []

    cat_cols: List[str] = []
    num_cols: List[str] = []
    for name, _, cols in preprocessor.transformers:
        if name == "cat":
            cat_cols.extend(list(cols))
        elif name == "num":
            num_cols.extend(list(cols))

    return cat_cols, num_cols


def _ensure_columns(df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> pd.DataFrame:
    """
    Fill any columns the model expects that are missing from the incoming request.
    'No' is a safe neutral default for categoricals (service not subscribed).
    0 is a safe default for numerics (zero spend, zero tenure).
    Called TWICE per request:
      1. Before add_features() -- ensures raw input columns exist
      2. After  add_features() -- ensures engineered columns exist
    """
    df = df.copy()
    for col in cat_cols:
        if col not in df.columns:
            df[col] = "No"
    for col in num_cols:
        if col not in df.columns:
            df[col] = 0
    return df


@app.get("/")
def home():
    return {"message": "Customer Churn Analytics API is running"}


@app.get("/health")
def health():
    # "ok" if model loaded at startup; "model_not_loaded" if .joblib was missing
    return {"status": "ok" if model is not None else "model_not_loaded"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded -- run src/train.py first")

    try:
        # Step 1: Convert Pydantic models -> DataFrame (one row per customer)
        records_df = pd.DataFrame([r.dict() for r in request.records])

        # Step 2: Save customer IDs NOW, before column selection drops them
        customer_ids = records_df.get(
            "customerID", pd.Series(range(len(records_df)))
        ).tolist()

        # Step 3: Get the exact column lists the preprocessor was trained on
        cat_cols, num_cols = _extract_columns()

        # Step 4: Fill missing raw input columns with safe defaults
        df = _ensure_columns(records_df, cat_cols, num_cols)

        # Step 5: Compute all engineered features (tenure, RFM, engagement, complaint_proxy)
        df = add_features(df)

        # Step 6: Fill any missing engineered columns (if add_features skipped one)
        df = _ensure_columns(df, cat_cols, num_cols)

        # Step 7: Select only expected columns in the correct order
        # Passing extra or reordered columns causes silent mispredictions
        expected_cols = cat_cols + num_cols
        if expected_cols:
            df = df[expected_cols]

        # Step 8: Run inference -- [:, 1] is P(churn=Yes)
        proba = model.predict_proba(df)[:, 1]
        preds = (proba >= 0.5).astype(int)

        # Step 9: Build one response item per customer
        results: List[PredictResponseItem] = []
        for i, row in df.iterrows():
            insights = generate_insights(row.to_dict(), float(proba[i]))
            results.append(
                PredictResponseItem(
                    customerID=str(customer_ids[i]),   # from saved IDs -- not from trimmed df
                    churn_probability=float(proba[i]),
                    churn_label="Yes" if preds[i] == 1 else "No",
                    insights=insights["insights"],
                )
            )

        return PredictResponse(predictions=results)

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
