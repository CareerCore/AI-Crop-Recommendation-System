"""
Crop Recommendation API + command-line test mode.

Run from the repository root:
    uvicorn ml.src.crop_api:app --reload

Swagger UI:
    http://127.0.0.1:8000/docs

Use the same module without a server:
    python ml/src/crop_api.py --once "{\"P\":42,\"humidity\":82,\"rainfall\":202.9}"

The endpoint loads the existing model bundle produced by train_all.py:
    ml/artifacts/crop_recommendation/crop/model.joblib

This file does not train a model. It only loads the trained model and predicts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Make imports work both when launched as a module (uvicorn) and as a script.
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import load_bundle, predict_crop

MODEL_PATH = (
    PROJECT_ROOT / "ml" / "artifacts" / "model.joblib"
)

app = FastAPI(
    title="Crop Recommendation API",
    version="1.0.0",
    description=(
        "Predicts the top crop classes from seven soil and climate inputs. "
        "Predictions are model outputs, not a guarantee of agronomic suitability."
    ),
)

# Lazy-loaded once per process. The first /predict request loads the bundle;
# subsequent requests reuse it rather than reading the file on every request.
_model_bundle: dict[str, Any] | None = None


class CropInput(BaseModel):
    # Units must match the training dataset's definitions and units.
    N: float = Field(..., ge=0, description="Nitrogen measurement")
    P: float = Field(..., ge=0, description="Phosphorus measurement")
    K: float = Field(..., ge=0, description="Potassium measurement")
    temperature: float = Field(..., description="Temperature in dataset units")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    ph: float = Field(..., ge=0, le=14, description="Soil pH")
    rainfall: float = Field(..., ge=0, description="Rainfall in dataset units")
    top_k: int = Field(default=3, ge=1, le=22, description="Number of suggestions")


class CropPrediction(BaseModel):
    label: str
    confidence: float


class CropOutput(BaseModel):
    predictions: list[CropPrediction]
    model_name: str | None = None
    model_version: str | None = None
    trained_rows: int | None = None
    note: str


def get_bundle() -> dict[str, Any]:
    """Load the model bundle on first use and reuse it for later requests."""
    global _model_bundle

    if _model_bundle is None:
        try:
            _model_bundle = load_bundle(MODEL_PATH)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Trained model not found at {MODEL_PATH}. "
                    "Run the training pipeline first."
                ),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not load the model bundle: {exc}",
            ) from exc

    return _model_bundle


@app.get("/health")
def health() -> dict[str, str]:
    """Simple service check; it does not guarantee that the model file exists."""
    return {"status": "ok", "service": "crop-recommendation"}


@app.post("/predict", response_model=CropOutput)
def predict(payload: CropInput) -> CropOutput:
    """
    Predict top-k crops for one observation.

    FastAPI validates the JSON fields and numeric limits before this function
    calls the shared inference.predict_crop() function.
    """
    bundle = get_bundle()

    try:
        results = predict_crop(
            bundle,
            N=payload.N,
            P=payload.P,
            K=payload.K,
            temperature=payload.temperature,
            humidity=payload.humidity,
            ph=payload.ph,
            rainfall=payload.rainfall,
            top_k=payload.top_k,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return CropOutput(
        predictions=results,
        model_name=bundle.get("model_name"),
        model_version=bundle.get("version"),
        trained_rows=bundle.get("trained_rows"),
        note=(
            "Confidence is the model's predicted class probability. "
            "It is not a guarantee of crop suitability, yield or farming success."
        ),
    )


def predict_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Call the same prediction logic directly from Python, without HTTP.

    Example:
        result = predict_record({
            "N": 90, "P": 42, "K": 43,
            "temperature": 20.8, "humidity": 82,
            "ph": 6.5, "rainfall": 202.9, "top_k": 3
        })
    """
    payload = CropInput(**record)
    bundle = get_bundle()
    predictions = predict_crop(
        bundle,
        N=payload.N,
        P=payload.P,
        K=payload.K,
        temperature=payload.temperature,
        humidity=payload.humidity,
        ph=payload.ph,
        rainfall=payload.rainfall,
        top_k=payload.top_k,
    )
    return {
        "predictions": predictions,
        "model_name": bundle.get("model_name"),
        "model_version": bundle.get("version"),
        "trained_rows": bundle.get("trained_rows"),
    }


def main() -> None:
    """Optional one-record command-line mode; the HTTP API remains available."""
    parser = argparse.ArgumentParser(description="Test the crop model once.")
    parser.add_argument(
        "--once",
        type=str,
        help="One JSON object containing the seven crop inputs and optional top_k.",
    )
    args = parser.parse_args()

    if args.once:
        record = json.loads(args.once)
        print(json.dumps(predict_record(record), indent=2))
        return

    import uvicorn

    uvicorn.run("ml.src.crop_api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
