"""
api.py  —  2B (cloud version)
==============================
FastAPI yang menjadi middleware antara Streamlit dan SageMaker endpoint.
Frontend (frontend.py) tetap manggil http://localhost:8000/predict seperti biasa.
Hanya file ini yang tahu soal SageMaker — inilah decoupled architecture-nya.

Environment variables:
  ENDPOINT_NAME   nama SageMaker endpoint  (default: credit-score-endpoint)
  AWS_REGION      region AWS               (default: us-east-1)
"""
import json
import os
from typing import Dict

import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
AWS_REGION    = os.environ.get("AWS_REGION",    "us-east-1")


class PredictRequest(BaseModel):
    Month:                    str
    Age:                      float
    Occupation:               str
    Annual_Income:            float
    Monthly_Inhand_Salary:    float
    Num_Bank_Accounts:        float
    Num_Credit_Card:          float
    Interest_Rate:            float
    Num_of_Loan:              float
    Delay_from_due_date:      float
    Num_of_Delayed_Payment:   float
    Changed_Credit_Limit:     float
    Num_Credit_Inquiries:     float
    Credit_Mix:               str
    Outstanding_Debt:         float
    Credit_Utilization_Ratio: float
    Payment_of_Min_Amount:    str
    Total_EMI_per_month:      float
    Amount_invested_monthly:  float
    Monthly_Balance:          float
    Payment_Behaviour:        str
    Credit_History_Months:    float
    Loan_Student_Loan:            int = 0
    Loan_Home_Equity_Loan:        int = 0
    Loan_Mortgage_Loan:           int = 0
    Loan_Debt_Consolidation_Loan: int = 0
    Loan_Credit_Builder_Loan:     int = 0
    Loan_Payday_Loan:             int = 0
    Loan_Personal_Loan:           int = 0
    Loan_Auto_Loan:               int = 0


class PredictResponse(BaseModel):
    label:         str
    probabilities: Dict[str, float]
    classes:       list[str]


class CreditScoreCloudAPI:
    """OOP wrapper untuk FastAPI — konsisten dengan arsitektur 1C/backend.py."""

    def __init__(self) -> None:
        self.app        = FastAPI(title="Credit Score API — Cloud")
        self.sm_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
        self._register_routes()

    def _register_routes(self) -> None:
        self.app.get("/health")(self.health)
        self.app.post("/predict", response_model=PredictResponse)(self.predict)

    def health(self) -> dict:
        return {"status": "ok", "endpoint": ENDPOINT_NAME}

    def predict(self, payload: PredictRequest) -> dict:
        try:
            body = json.dumps(payload.model_dump())
            resp = self.sm_runtime.invoke_endpoint(
                EndpointName = ENDPOINT_NAME,
                ContentType  = "application/json",
                Body         = body,
            )
            return json.loads(resp["Body"].read().decode())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


app = CreditScoreCloudAPI().app
