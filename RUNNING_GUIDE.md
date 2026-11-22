# Running Guide – AI Crop Recommendation System

This guide explains how to prepare the dataset, train the machine-learning models, and run the API with interactive Swagger documentation.

## 1. Project Structure

Before running the project, make sure your directory structure looks like this:

```text
AI-Crop-Recommendation-System/
│
├── ml/
│   ├── data/
│   │   └── raw/
│   │       └── Crop_recommendation.csv
│   │
│   ├── src/
│   │   ├── train_all.py
│   │   ├── inference.py
│   │   └── crop_api.py
│   │
│   ├── artifacts/
│   └── reports/
│
├── README.md
├── API_README.md
├── CONTRIBUTING.md
├── SUPPORT.md
└── RUNNING_GUIDE.md
```

## 2. Dataset Setup

Place the training dataset in the following directory:

```text
ml/data/raw/Crop_recommendation.csv
```

**Important:**
- Keep the filename as `Crop_recommendation.csv`, unless you also update the dataset path in the training script.
- The CSV file should contain the required input features and target column.

The expected columns are:

| Column | Description |
|---|---|
| N | Nitrogen content |
| P | Phosphorus content |
| K | Potassium content |
| temperature | Temperature |
| humidity | Relative humidity |
| ph | Soil pH |
| rainfall | Rainfall |
| label | Recommended crop |

The `label` column contains the crop names used for model training.

## 3. Install Dependencies

Open a terminal in the project's root directory.

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

If your project uses separate dependency files, install the appropriate machine-learning and API dependencies instead.

## 4. Train the Machine-Learning Models

Run the training script from the project root:

```bash
python ml/src/train_all.py
```

The training pipeline performs tasks such as:

- Loading and preprocessing the dataset.
- Feature engineering.
- Splitting the dataset into training and testing sets.
- Training and tuning multiple machine-learning algorithms.
- Evaluating models using classification metrics.
- Comparing model performance.
- Selecting the final model.
- Generating evaluation reports and visualizations.
- Saving the trained model artifact.

### Generated Model

After successful training, the model artifact is saved in:

```text
ml/artifacts/model.joblib
```

The API uses this saved artifact for predictions. You do not need to retrain the model every time you start the API.

### Training Reports

Training reports and evaluation visualizations are generated under:

```text
ml/reports/
```

These may include:
- Model comparison results.
- Classification reports.
- Confusion matrices.
- Feature importance visualizations.
- Evaluation metrics.

## 5. Run the FastAPI Server

Once model training has completed successfully, start the API from the project root:

```bash
uvicorn ml.src.crop_api:app --reload
```

The API will be available locally at:

```text
http://127.0.0.1:8000
```

## 6. Open Swagger Documentation

Open the following URL in your browser:

**Swagger UI:**

```text
http://127.0.0.1:8000/docs
```

Swagger UI provides an interactive interface to:

- View available API endpoints.
- Inspect request and response schemas.
- Submit crop prediction requests.
- Test the prediction endpoint.
- View API responses directly in your browser.

You can also access the alternative ReDoc documentation at:

```text
http://127.0.0.1:8000/redoc
```

## 7. Test the Crop Prediction API

In Swagger UI:

1. Find the `POST /predict` endpoint.
2. Expand the endpoint.
3. Click **Try it out**.
4. Enter the required soil and environmental parameters.
5. Click **Execute**.
6. Review the recommended crop and prediction results.

Example request:

```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.8,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9,
  "top_k": 3
}
```

The API returns crop recommendations based on the supplied input parameters.

## 8. Important Notes

- Run commands from the project's root directory.
- Make sure the CSV dataset exists before starting model training.
- Train the model before starting the API if the required model artifact does not exist.
- The API loads the saved model artifact; it does not train the model when making predictions.
- Keep the training and inference feature-engineering logic consistent.
- Use a compatible Python environment with all required dependencies installed.
- The dataset's predictions should be validated against local agricultural conditions before being used for real-world farming decisions.

---

**Quick Commands**

| Task | Command |
|---|---|
| Train models | `python ml/src/train_all.py` |
| Start API | `uvicorn ml.src.crop_api:app --reload` |
| Open Swagger | `http://127.0.0.1:8000/docs` |
| Open ReDoc | `http://127.0.0.1:8000/redoc` |