import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, RobustScaler

DROP_COLS = ["ID", "Name", "SSN", "Customer_ID"]

STR_TO_FLOAT_COLS = [
    "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly", "Monthly_Balance",
]

OUTLIER_CAPS = {
    "Annual_Income": 0.98,
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

NUM_COLS = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
    "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Total_EMI_per_month", "Amount_invested_monthly",
    "Monthly_Balance", "Credit_History_Months",
]

CAT_COLS = [
    "Month", "Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour",
]


def _clean_str_to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce",
    )


def _parse_credit_history_months(val) -> float:
    nums = re.findall(r"\d+", str(val))
    if len(nums) == 2:
        return int(nums[0]) * 12 + int(nums[1])
    return np.nan


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless row-level feature engineering applied before train/test split."""
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
    df["Credit_Mix"] = df["Credit_Mix"].apply(lambda x: "Unknown" if x == "_" or pd.isna(x) else x)
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


class Preprocessor:

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.label_encoder = LabelEncoder()
        self.column_transformer: ColumnTransformer | None = None

    def _build_column_transformer(self, X: pd.DataFrame) -> ColumnTransformer:
        num_cols = [c for c in NUM_COLS if c in X.columns]
        cat_cols = [c for c in CAT_COLS if c in X.columns]

        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])

        return ColumnTransformer([
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ], remainder="passthrough")

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        self.column_transformer = self._build_column_transformer(X_train)
        X_train_processed = self.column_transformer.fit_transform(X_train)
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        return X_train_processed, y_train_encoded

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.column_transformer.transform(X)

    def transform_target(self, y: pd.Series) -> np.ndarray:
        return self.label_encoder.transform(y)

    def save(self, path: str | Path) -> None:
        """Save the fitted label encoder (kept identical to the original label_encoder.pkl)."""
        with open(path, "wb") as f:
            pickle.dump(self.label_encoder, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self.label_encoder = pickle.load(f)
