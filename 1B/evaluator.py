from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from trainer import BaseTrainer


class Evaluator:
    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names

    def evaluate(
        self,
        trainer: BaseTrainer,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        y_train_pred = trainer.predict(X_train)
        y_test_pred = trainer.predict(X_test)

        report = classification_report(
            y_test, y_test_pred,
            target_names=self.class_names,
            output_dict=True,
        )

        result = {
            "model": trainer.name,
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "f1_macro": f1_score(y_test, y_test_pred, average="macro"),
            "report": report,
        }
        if hasattr(trainer, "best_params_"):
            result["best_params"] = trainer.best_params_
        return result

    def print_report(self, results: list[dict]) -> None:
        print(f"\n{'Model':<20}{'Train Acc':>12}{'Test Acc':>12}{'F1 Macro':>12}")
        print("-" * 56)
        for r in results:
            print(
                f"{r['model']:<20}{r['train_accuracy']:>12.4f}"
                f"{r['test_accuracy']:>12.4f}{r['f1_macro']:>12.4f}"
            )

    def select_best(self, results: list[dict], metric: str = "f1_macro") -> dict:
        return max(results, key=lambda r: r[metric])

    def save_evaluation(self, results: list[dict], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        best = self.select_best(results)

        with open(path, "w") as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Macro F1: {best['f1_macro']:.4f}\n\n")

            f.write("Model Comparison:\n")
            for r in results:
                f.write(
                    f"  {r['model']:<15} train_accuracy={r['train_accuracy']:.4f}, "
                    f"test_accuracy={r['test_accuracy']:.4f}, f1_macro={r['f1_macro']:.4f}\n"
                )
            f.write(f"\nBest model: {best['model']}\n")
            if "best_params" in best:
                f.write(f"Best params: {best['best_params']}\n")
            f.write("\n")

            f.write("Per-class metrics (best model):\n")
            for cls, vals in best["report"].items():
                if isinstance(vals, dict):
                    f.write(
                        f"  {cls}: precision={vals['precision']:.4f}, "
                        f"recall={vals['recall']:.4f}, f1={vals['f1-score']:.4f}\n"
                    )
