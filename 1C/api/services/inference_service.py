import pickle
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE / "model.pkl"
LE_PATH = BASE / "label_encoder.pkl"


class InferenceService:
    def __init__(self):
        with open(MODEL_PATH, "rb") as f:
            self.pipeline = pickle.load(f)
        with open(LE_PATH, "rb") as f:
            self.le = pickle.load(f)

    def predict(self, input_dict: dict) -> dict:
        df_input = pd.DataFrame([input_dict])
        pred_encoded = self.pipeline.predict(df_input)
        pred_proba = self.pipeline.predict_proba(df_input)[0]
        label = self.le.inverse_transform(pred_encoded)[0]

        probabilities = {
            cls: float(prob) for cls, prob in zip(self.le.classes_, pred_proba)
        }

        return {
            "label": label,
            "probabilities": probabilities,
            "classes": list(self.le.classes_),
        }
