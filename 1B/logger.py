import pickle
from datetime import datetime

import mlflow

from config import (
    EVAL_PATH, LABEL_ENCODER_PATH, MODEL_PATH,
    MLRUNS_DIR, RF_PARAMS, TRAIN_PARAMS,
)


def get_existing_f1():
    """Read macro_f1 from previous evaluation.txt if model.pkl exists."""
    if not MODEL_PATH.exists():
        return None
    if not EVAL_PATH.exists():
        return None
    with open(EVAL_PATH) as f:
        for line in f:
            if line.startswith("Macro F1:"):
                try:
                    return float(line.split(":")[1].strip())
                except ValueError:
                    pass
    return None


def save_evaluation(metrics):
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_PATH, "w") as f:
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n\n")
        f.write("Per-class metrics:\n")
        for cls, vals in metrics["report"].items():
            if isinstance(vals, dict):
                f.write(
                    f"  {cls}: precision={vals['precision']:.4f}, "
                    f"recall={vals['recall']:.4f}, f1={vals['f1-score']:.4f}\n"
                )


def save_artifacts(pipeline, label_encoder):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"[logger] Saved model       -> {MODEL_PATH}")
    print(f"[logger] Saved label enc   -> {LABEL_ENCODER_PATH}")


def log_to_mlflow(metrics, existing_f1, model_saved):
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("credit_score_pipeline")

    with mlflow.start_run():
        mlflow.log_param("n_estimators", RF_PARAMS["n_estimators"])
        mlflow.log_param("test_size", TRAIN_PARAMS["test_size"])
        mlflow.log_param("random_state", TRAIN_PARAMS["random_state"])

        mlflow.log_metric("macro_f1", metrics["macro_f1"])
        for cls, vals in metrics["report"].items():
            if isinstance(vals, dict):
                safe = cls.replace(" ", "_")
                mlflow.log_metric(f"{safe}_f1", vals["f1-score"])
                mlflow.log_metric(f"{safe}_precision", vals["precision"])
                mlflow.log_metric(f"{safe}_recall", vals["recall"])

        if existing_f1 is not None:
            is_better = metrics["macro_f1"] >= existing_f1
            mlflow.log_metric("existing_macro_f1", existing_f1)
            mlflow.log_param("better_than_existing", is_better)
            print(
                f"[logger] Existing macro_f1={existing_f1:.4f} | "
                f"New={metrics['macro_f1']:.4f} | Better={is_better}"
            )
        else:
            mlflow.log_param("better_than_existing", "first_run")
            print("[logger] No existing model - first run.")

        mlflow.log_param("model_saved", model_saved)

        if EVAL_PATH.exists():
            mlflow.log_artifact(str(EVAL_PATH))
        if model_saved and MODEL_PATH.exists():
            mlflow.log_artifact(str(MODEL_PATH))
