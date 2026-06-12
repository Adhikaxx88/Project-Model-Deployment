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


def load_data(path=None):
    if path is None:
        path = DATA_PATH
    df = pd.read_csv(path, index_col=0)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df
