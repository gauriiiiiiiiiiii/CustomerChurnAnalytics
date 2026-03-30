from pydantic import BaseModel
from typing import List, Optional


class CustomerRecord(BaseModel):
    customerID: str
    gender: Optional[str] = None
    SeniorCitizen: int
    Partner: Optional[str] = None
    Dependents: Optional[str] = None
    tenure: float
    PhoneService: Optional[str] = None
    MultipleLines: Optional[str] = None
    InternetService: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    Contract: Optional[str] = None
    PaperlessBilling: Optional[str] = None
    PaymentMethod: Optional[str] = None
    MonthlyCharges: float
    TotalCharges: Optional[float] = None


class PredictRequest(BaseModel):
    records: List[CustomerRecord]


class PredictResponseItem(BaseModel):
    customerID: str
    churn_probability: float
    churn_label: str
    insights: List[str]


class PredictResponse(BaseModel):
    predictions: List[PredictResponseItem]
