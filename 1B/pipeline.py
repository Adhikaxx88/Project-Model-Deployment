import data_loader
import evaluator
import logger
import preprocessor
import trainer


def main():
    print("[pipeline] Loading data...")
    df = data_loader.load_data()

    print("[pipeline] Preprocessing...")
    df = preprocessor.preprocess(df)

    print("[pipeline] Training...")
    pipeline, le, X_test, y_test = trainer.train(df)

    print("[pipeline] Evaluating...")
    metrics = evaluator.evaluate(pipeline, le, X_test, y_test)
    print(f"[pipeline] Macro F1: {metrics['macro_f1']:.4f}")

    existing_f1 = logger.get_existing_f1()

    logger.save_evaluation(metrics)

    should_save = (existing_f1 is None) or (metrics["macro_f1"] >= existing_f1)
    if should_save:
        logger.save_artifacts(pipeline, le)
    else:
        print(
            f"[pipeline] New model ({metrics['macro_f1']:.4f}) did not beat "
            f"existing ({existing_f1:.4f}) - skipping save."
        )

    print("[pipeline] Logging to MLflow...")
    logger.log_to_mlflow(metrics, existing_f1, should_save)

    print("[pipeline] Done.")


if __name__ == "__main__":
    main()
