from sklearn.metrics import classification_report, f1_score


def evaluate(pipeline, label_encoder, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
    )
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    return {"macro_f1": macro_f1, "report": report}
