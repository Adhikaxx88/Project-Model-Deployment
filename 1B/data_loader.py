from pathlib import Path

import pandas as pd

from config import DATA_PATH

REQUIRED_COLS = [
    "ID", "Customer_ID", "Name", "Age", "SSN", "Occupation",
    "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Type_of_Loan",
    "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
    "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Credit_History_Age", "Payment_of_Min_Amount",
    "Total_EMI_per_month", "Amount_invested_monthly", "Payment_Behaviour",
    "Monthly_Balance", "Credit_Score",
]


class DataLoader:
    REQUIRED_COLS = REQUIRED_COLS

    def __init__(self, filepath: str | Path | None = None) -> None:
        self.filepath = Path(filepath) if filepath is not None else DATA_PATH

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath, index_col=0)
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df
