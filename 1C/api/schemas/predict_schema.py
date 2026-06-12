from typing import Dict

from pydantic import BaseModel


class PredictRequest(BaseModel):
    Month: str
    Age: int
    Occupation: str
    Monthly_Inhand_Salary: float
    Num_Bank_Accounts: int
    Num_Credit_Card: int
    Interest_Rate: int
    Num_of_Loan: int
    Delay_from_due_date: int
    Num_of_Delayed_Payment: int
    Changed_Credit_Limit: float
    Num_Credit_Inquiries: int
    Credit_Mix: str
    Outstanding_Debt: float
    Credit_Utilization_Ratio: float
    Payment_of_Min_Amount: str
    Total_EMI_per_month: float
    Amount_invested_monthly: float
    Payment_Behaviour: str
    Credit_History_Months: int
    Loan_Student_Loan: int
    Loan_Home_Equity_Loan: int
    Loan_Mortgage_Loan: int
    Loan_Debt_Consolidation_Loan: int
    Loan_Credit_Builder_Loan: int
    Loan_Payday_Loan: int
    Loan_Personal_Loan: int
    Loan_Auto_Loan: int


class PredictResponse(BaseModel):
    label: str
    probabilities: Dict[str, float]
    classes: list[str]
