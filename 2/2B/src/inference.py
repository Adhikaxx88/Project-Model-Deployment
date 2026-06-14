
import json
import os
import pickle

import pandas as pd

NUM_COLS = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance", "Credit_History_Months",
]


def model_fn(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    return {"model": model, "label_encoder": label_encoder}


def input_fn(request_body: str, content_type: str) -> pd.DataFrame:
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    data = json.loads(request_body)
    df = pd.DataFrame([data])
    for col in NUM_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def predict_fn(input_data: pd.DataFrame, model_dict: dict) -> dict:
    model = model_dict["model"]
    le    = model_dict["label_encoder"]
    proba        = model.predict_proba(input_data)[0]
    pred_encoded = int(model.predict(input_data)[0])
    pred_label   = str(le.inverse_transform([pred_encoded])[0])
    classes      = list(le.classes_)
    return {
        "label":         pred_label,
        "probabilities": {cls: float(p) for cls, p in zip(classes, proba)},
        "classes":       classes,
    }


def output_fn(prediction: dict, accept: str) -> str:
    return json.dumps(prediction)
