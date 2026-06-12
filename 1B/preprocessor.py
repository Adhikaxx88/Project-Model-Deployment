import re

import numpy as np
import pandas as pd

DROP_COLS = ["ID", "Name", "SSN", "Customer_ID"]

STR_TO_FLOAT_COLS = [
    "Age", "Num_of_Loan", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly",
]

OUTLIER_CAPS = {
    "Annual_Income": 0.98,
    "Monthly_Balance": 0.98,
    "Num_Bank_Accounts": 0.98,
    "Num_Credit_Card": 0.95,
    "Interest_Rate": 0.95,
    "Num_of_Loan": 0.95,
    "Num_Credit_Inquiries": 0.98,
    "Num_of_Delayed_Payment": 0.99,
    "Total_EMI_per_month": 0.95,
}

LOAN_TYPES = [
    "Student Loan", "Home Equity Loan", "Mortgage Loan",
    "Debt Consolidation Loan", "Credit-Builder Loan",
    "Payday Loan", "Personal Loan", "Auto Loan",
]


def _clean_str_to_float(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce",
    )


def _parse_credit_history_months(val):
    nums = re.findall(r"\d+", str(val))
    if len(nums) == 2:
        return int(nums[0]) * 12 + int(nums[1])
    return np.nan


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    for col in STR_TO_FLOAT_COLS:
        if col in df.columns:
            df[col] = _clean_str_to_float(df[col])

    median_age = df.loc[df["Age"] <= 100, "Age"].median()
    df.loc[df["Age"] > 100, "Age"] = median_age
    df["Num_Bank_Accounts"] = df["Num_Bank_Accounts"].abs()
    df["Delay_from_due_date"] = df["Delay_from_due_date"].abs()

    df["Occupation"] = df["Occupation"].replace("_______", "Unknown")
    df["Credit_Mix"] = df["Credit_Mix"].replace("_", "Unknown")
    df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", "Unknown")
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace("NM", "Unknown")

    for loan in LOAN_TYPES:
        col_name = f"Loan_{loan.replace(' ', '_').replace('-', '_')}"
        df[col_name] = df["Type_of_Loan"].fillna("").apply(lambda x, l=loan: 1 if l in x else 0)
    df.drop(columns=["Type_of_Loan"], inplace=True)

    df["Credit_History_Months"] = df["Credit_History_Age"].apply(_parse_credit_history_months)
    df.drop(columns=["Credit_History_Age"], inplace=True)

    for col, q in OUTLIER_CAPS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            cap = df[col].quantile(q)
            df.loc[df[col] > cap, col] = cap

    return df
