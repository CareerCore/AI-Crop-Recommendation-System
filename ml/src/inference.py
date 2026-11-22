from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def add_crop_features(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering with no fitted state (cannot leak). Used at training AND serving."""
    X = X[CROP_FEATURES].astype(float).copy()
    X["NPK_sum"] = X["N"] + X["P"] + X["K"]
    X["N_to_P"] = X["N"] / (X["P"] + 1)
    X["N_to_K"] = X["N"] / (X["K"] + 1)
    X["P_to_K"] = X["P"] / (X["K"] + 1)
    X["THI"] = X["temperature"] * X["humidity"] / 100
    X["ph_dev"] = (X["ph"] - 6.5).abs()
    X["log_rain"] = np.log1p(X["rainfall"])
    return X


def load_bundle(path) -> dict:
    return joblib.load(path)


def _top(bundle: dict, proba: np.ndarray, k: int) -> list[dict]:
    best = np.argsort(proba)[::-1][:k]
    return [
        {"label": bundle["classes"][i], "confidence": round(float(proba[i]), 4)}
        for i in best
    ]


def predict_crop(
    bundle: dict, N, P, K, temperature, humidity, ph, rainfall, top_k: int = 3
) -> list[dict]:
    """Top-k crop recommendations with probabilities."""
    row = pd.DataFrame(
        [
            {
                "N": N,
                "P": P,
                "K": K,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
            }
        ]
    )
    proba = bundle["model"].predict_proba(add_crop_features(row))[0]
    return _top(bundle, proba, top_k)


def predict_bug(bundle: dict, top_k: int = 3, **fields) -> dict:
    """
    Predict one label (severity or priority) for a bug report.
    Pass the report's fields by their original column names, e.g.
        predict_bug(bundle, summary="Crash on startup", product="Firefox", component="General")
    Fields the model does not use are ignored; missing categorical fields count as unknown.
    """
    text = " . ".join(str(fields.get(c) or "") for c in bundle["text_cols"])
    row = {"text": text, **{c: str(fields.get(c) or "NA") for c in bundle["cat_cols"]}}
    proba = bundle["model"].predict_proba(pd.DataFrame([row]))[0]
    top = _top(bundle, proba, top_k)
    return {
        "label": top[0]["label"],
        "confidence": top[0]["confidence"],
        "alternatives": top[1:],
        "model_version": bundle["version"],
    }
