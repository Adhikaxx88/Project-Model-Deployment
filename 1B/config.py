from pathlib import Path

BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "data_A (1).csv"

OUTPUTS_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUTPUTS_DIR / "model.pkl"
LABEL_ENCODER_PATH = OUTPUTS_DIR / "label_encoder.pkl"
EVAL_PATH = OUTPUTS_DIR / "evaluation.txt"

MLRUNS_DIR = BASE_DIR / "mlruns"

RF_PARAMS = {
    "n_estimators": 200,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

TRAIN_PARAMS = {
    "test_size": 0.2,
    "random_state": 42,
}
