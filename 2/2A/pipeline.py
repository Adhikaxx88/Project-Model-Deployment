import pickle

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import EVAL_PATH, LABEL_ENCODER_PATH, MLFLOW_EXPERIMENT_NAME, MLRUNS_DIR, MODEL_PATH, TRAIN_PARAMS
from data_loader import DataLoader
from evaluator import Evaluator
from preprocessor import Preprocessor, engineer_features
from trainer import get_all_trainers, get_tuned_trainers


class TrainingOrchestrator:
    def __init__(self) -> None:
        self.loader = DataLoader()
        self.preprocessor = Preprocessor()

    def run(self) -> None:
        print("[pipeline] Loading data...")
        df = self.loader.load()

        print("[pipeline] Preprocessing...")
        df = engineer_features(df)

        X = df.drop("Credit_Score", axis=1)
        y = df["Credit_Score"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TRAIN_PARAMS["test_size"],
            random_state=TRAIN_PARAMS["random_state"],
            stratify=y,
        )

        X_train_proc, y_train_enc = self.preprocessor.fit_transform(X_train, y_train)
        X_test_proc = self.preprocessor.transform(X_test)
        y_test_enc = self.preprocessor.transform_target(y_test)

        evaluator = Evaluator(class_names=list(self.preprocessor.label_encoder.classes_))

        mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        results = []
        for trn in get_all_trainers():
            print(f"[pipeline] Training {trn.name}...")

            with mlflow.start_run(run_name=trn.name):
                trn.train(X_train_proc, y_train_enc)
                metrics = evaluator.evaluate(trn, X_train_proc, y_train_enc, X_test_proc, y_test_enc)

                mlflow.log_params({
                    "model": trn.name,
                    "test_size": TRAIN_PARAMS["test_size"],
                    "random_state": TRAIN_PARAMS["random_state"],
                })
                mlflow.log_metrics({
                    "train_accuracy": metrics["train_accuracy"],
                    "test_accuracy": metrics["test_accuracy"],
                    "f1_macro": metrics["f1_macro"],
                })

            results.append({**metrics, "trainer": trn})

        for trn in get_tuned_trainers():
            print(f"[pipeline] Tuning {trn.name}...")

            with mlflow.start_run(run_name=trn.name):
                trn.train(X_train_proc, y_train_enc)
                metrics = evaluator.evaluate(trn, X_train_proc, y_train_enc, X_test_proc, y_test_enc)
                print(f"[pipeline] {trn.name} best params: {trn.best_params_}")

                mlflow.log_params({
                    "model": trn.name,
                    "test_size": TRAIN_PARAMS["test_size"],
                    "random_state": TRAIN_PARAMS["random_state"],
                    **trn.best_params_,
                })
                mlflow.log_metrics({
                    "train_accuracy": metrics["train_accuracy"],
                    "test_accuracy": metrics["test_accuracy"],
                    "f1_macro": metrics["f1_macro"],
                })

            results.append({**metrics, "trainer": trn})

        evaluator.print_report(results)

        best = evaluator.select_best(results)
        best_trainer = best["trainer"]
        print(f"\n[pipeline] Best model: {best_trainer.name} (f1_macro={best['f1_macro']:.4f})")

        evaluator.save_evaluation(results, EVAL_PATH)

        self.preprocessor.save(LABEL_ENCODER_PATH)
        print(f"[pipeline] Saved label encoder -> {LABEL_ENCODER_PATH}")

        best_pipeline = Pipeline([
            ("preprocessor", self.preprocessor.column_transformer),
            ("classifier", best_trainer.final_estimator),
        ])

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(best_pipeline, f)
        print(f"[pipeline] Saved model -> {MODEL_PATH}")

        with mlflow.start_run(run_name=f"best_{best_trainer.name}"):
            mlflow.log_params({
                "model": best_trainer.name,
                "test_size": TRAIN_PARAMS["test_size"],
            })
            mlflow.log_metrics({
                "train_accuracy": best["train_accuracy"],
                "test_accuracy": best["test_accuracy"],
                "f1_macro": best["f1_macro"],
            })
            mlflow.sklearn.log_model(best_pipeline, "best_model")

        print("[pipeline] Done.")


if __name__ == "__main__":
    TrainingOrchestrator().run()
