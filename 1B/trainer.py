from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, RobustScaler

from config import RF_PARAMS, TRAIN_PARAMS

NUM_COLS = [
    "Age", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
    "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Total_EMI_per_month", "Amount_invested_monthly",
    "Credit_History_Months",
]

CAT_COLS = [
    "Month", "Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour",
]


def train(df):
    X = df.drop("Credit_Score", axis=1)
    y = df["Credit_Score"]

    loan_cols = [c for c in X.columns if c.startswith("Loan_")]
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

    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(**RF_PARAMS)),
    ])

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=TRAIN_PARAMS["test_size"],
        random_state=TRAIN_PARAMS["random_state"],
        stratify=y_enc,
    )

    pipeline.fit(X_train, y_train)
    return pipeline, le, X_test, y_test
