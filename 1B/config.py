from pathlib import Path

BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "data_A (1).csv"

OUTPUTS_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUTPUTS_DIR / "model.pkl"
LABEL_ENCODER_PATH = OUTPUTS_DIR / "label_encoder.pkl"
EVAL_PATH = OUTPUTS_DIR / "evaluation.txt"

MLRUNS_DIR = BASE_DIR / "mlruns"

MLFLOW_EXPERIMENT_NAME = "credit_score_pipeline_model_v2"

RF_PARAMS = {
    "n_estimators": 200,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 6,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
}

LGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 6,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

MLP_PARAMS = {
    "hidden_layer_sizes": (256, 128, 64),
    "activation": "relu",
    "alpha": 0.001,
    "learning_rate_init": 0.001,
    "max_iter": 300,
    "random_state": 42,
}

TRAIN_PARAMS = {
    "test_size": 0.2,
    "random_state": 42,
}

SEARCH_PARAMS = {
    "n_iter": 20,
    "cv": 3,
    "scoring": "f1_macro",
    "random_state": 42,
    "n_jobs": -1,
}

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [None, 10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 8],
}

XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 400, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
}

LGBM_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 400, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
    "max_depth": [3, 4, 5, 6, 7, -1],
    "num_leaves": [15, 31, 50, 70, 100],
    "min_child_samples": [5, 10, 20, 30, 50],
}

MLP_PARAM_GRID = {
    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50), (100, 100)],
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.001, 0.01, 0.1],
    "learning_rate_init": [0.001, 0.01, 0.05, 0.1],
}
