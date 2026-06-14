
import json

import boto3
import sagemaker
from sagemaker.sklearn import SKLearnModel


BUCKET        = "credit-score-adhika-1234"
MODEL_S3_KEY  = "credit-score/model.tar.gz"
ENDPOINT_NAME = "credit-score-endpoint"
#

REGION = "us-east-1"

sess       = boto3.Session(region_name=REGION)
account_id = sess.client("sts").get_caller_identity()["Account"]
ROLE_ARN   = f"arn:aws:iam::{account_id}:role/LabRole"

sm_session = sagemaker.Session(boto_session=sess)

model = SKLearnModel(
    model_data        = f"s3://{BUCKET}/{MODEL_S3_KEY}",
    role              = ROLE_ARN,
    entry_point       = "inference.py",
    source_dir        = "src",
    framework_version = "1.2-1",
    py_version        = "py3",
    sagemaker_session = sm_session,
)

print("Deploying endpoint...")
predictor = model.deploy(
    initial_instance_count = 1,
    instance_type          = "ml.t3.medium",
    endpoint_name          = ENDPOINT_NAME,
)
print(f"Endpoint live: {ENDPOINT_NAME}")

sm_runtime = sess.client("sagemaker-runtime")

SAMPLES = {
    "Good (expected)": {
        "Month": "January", "Age": 35.0, "Occupation": "Doctor",
        "Annual_Income": 90000.0, "Monthly_Inhand_Salary": 7000.0,
        "Num_Bank_Accounts": 2.0, "Num_Credit_Card": 3.0,
        "Interest_Rate": 7.0, "Num_of_Loan": 1.0,
        "Delay_from_due_date": 2.0, "Num_of_Delayed_Payment": 1.0,
        "Changed_Credit_Limit": 2.0, "Num_Credit_Inquiries": 1.0,
        "Credit_Mix": "Good", "Outstanding_Debt": 200.0,
        "Credit_Utilization_Ratio": 10.0, "Payment_of_Min_Amount": "Yes",
        "Total_EMI_per_month": 50.0, "Amount_invested_monthly": 1000.0,
        "Monthly_Balance": 2000.0,
        "Payment_Behaviour": "High_spent_Large_value_payments",
        "Credit_History_Months": 120.0,
        "Loan_Student_Loan": 0, "Loan_Home_Equity_Loan": 0,
        "Loan_Mortgage_Loan": 1, "Loan_Debt_Consolidation_Loan": 0,
        "Loan_Credit_Builder_Loan": 0, "Loan_Payday_Loan": 0,
        "Loan_Personal_Loan": 0, "Loan_Auto_Loan": 0,
    },
    "Standard (expected)": {
        "Month": "March", "Age": 30.0, "Occupation": "Engineer",
        "Annual_Income": 40000.0, "Monthly_Inhand_Salary": 3000.0,
        "Num_Bank_Accounts": 3.0, "Num_Credit_Card": 4.0,
        "Interest_Rate": 13.0, "Num_of_Loan": 3.0,
        "Delay_from_due_date": 15.0, "Num_of_Delayed_Payment": 10.0,
        "Changed_Credit_Limit": 9.0, "Num_Credit_Inquiries": 4.0,
        "Credit_Mix": "Standard", "Outstanding_Debt": 800.0,
        "Credit_Utilization_Ratio": 28.0, "Payment_of_Min_Amount": "No",
        "Total_EMI_per_month": 120.0, "Amount_invested_monthly": 300.0,
        "Monthly_Balance": 300.0,
        "Payment_Behaviour": "Low_spent_Small_value_payments",
        "Credit_History_Months": 60.0,
        "Loan_Student_Loan": 0, "Loan_Home_Equity_Loan": 0,
        "Loan_Mortgage_Loan": 0, "Loan_Debt_Consolidation_Loan": 0,
        "Loan_Credit_Builder_Loan": 0, "Loan_Payday_Loan": 0,
        "Loan_Personal_Loan": 1, "Loan_Auto_Loan": 0,
    },
    "Poor (expected)": {
        "Month": "June", "Age": 22.0, "Occupation": "Unknown",
        "Annual_Income": 10000.0, "Monthly_Inhand_Salary": 800.0,
        "Num_Bank_Accounts": 8.0, "Num_Credit_Card": 9.0,
        "Interest_Rate": 30.0, "Num_of_Loan": 8.0,
        "Delay_from_due_date": 60.0, "Num_of_Delayed_Payment": 22.0,
        "Changed_Credit_Limit": 0.5, "Num_Credit_Inquiries": 15.0,
        "Credit_Mix": "Bad", "Outstanding_Debt": 3000.0,
        "Credit_Utilization_Ratio": 95.0, "Payment_of_Min_Amount": "No",
        "Total_EMI_per_month": 400.0, "Amount_invested_monthly": 10.0,
        "Monthly_Balance": 50.0,
        "Payment_Behaviour": "High_spent_Small_value_payments",
        "Credit_History_Months": 5.0,
        "Loan_Student_Loan": 1, "Loan_Home_Equity_Loan": 0,
        "Loan_Mortgage_Loan": 0, "Loan_Debt_Consolidation_Loan": 1,
        "Loan_Credit_Builder_Loan": 0, "Loan_Payday_Loan": 1,
        "Loan_Personal_Loan": 1, "Loan_Auto_Loan": 0,
    },
}

print("\n── Smoke tests ──")
for label, payload in SAMPLES.items():
    resp = sm_runtime.invoke_endpoint(
        EndpointName = ENDPOINT_NAME,
        ContentType  = "application/json",
        Body         = json.dumps(payload),
    )
    result = json.loads(resp["Body"].read().decode())
    print(f"{label:25s} → predicted: {result['label']:8s}  proba: {result['probabilities']}")
