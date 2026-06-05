# schemas.py
# Pydantic models that define the exact shape of data coming IN and going OUT of the API.
#
# Request:  CustomerRecord → PredictRequest
# Response: PredictResponseItem → PredictResponse
#
# Pydantic auto-validates field types on every request and raises HTTP 422 for bad input.
# Most CustomerRecord fields are Optional because the API fills missing ones with safe
# defaults ("No" / 0) before running feature engineering — see _ensure_columns() in api.py.

from pydantic import BaseModel
from typing import List, Optional


class CustomerRecord(BaseModel):
    # One customer's raw attributes — column names match the original training dataset.
    customerID: str
    gender: Optional[str] = None
    SeniorCitizen: int                      # 0 = Not a senior, 1 = Senior citizen
    Partner: Optional[str] = None           # "Yes" / "No"
    Dependents: Optional[str] = None        # "Yes" / "No"
    tenure: float                           # Months the customer has been with the company
    PhoneService: Optional[str] = None      # "Yes" / "No"
    MultipleLines: Optional[str] = None     # "Yes" / "No" / "No phone service"
    InternetService: Optional[str] = None   # "DSL" / "Fiber optic" / "No"
    OnlineSecurity: Optional[str] = None    # "Yes" / "No" / "No internet service"
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    Contract: Optional[str] = None          # "Month-to-month" / "One year" / "Two year"
    PaperlessBilling: Optional[str] = None  # "Yes" / "No"
    PaymentMethod: Optional[str] = None     # "Electronic check" / "Mailed check" / etc.
    MonthlyCharges: float                   # Recurring monthly bill amount
    TotalCharges: Optional[float] = None    # Cumulative charges; blank for new customers


class PredictRequest(BaseModel):
    # Wraps a list of CustomerRecords — supports single or batch prediction in one call.
    records: List[CustomerRecord]


class PredictResponseItem(BaseModel):
    # One prediction result returned per input customer.
    customerID: str
    churn_probability: float    # 0.0 to 1.0 — model's confidence the customer will churn
    churn_label: str            # "Yes" if probability >= 0.5, else "No"
    insights: List[str]         # Business action recommendations from insights.py


class PredictResponse(BaseModel):
    # Top-level response wrapper — one PredictResponseItem per input record.
    predictions: List[PredictResponseItem]
