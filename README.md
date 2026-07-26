# Streamlit and AWS Web Deployment — Credit Score Prediction

An end-to-end machine learning project for predicting credit score categories (Good/Standard/Poor), covering everything from data exploration and a structured training pipeline to deployment on AWS (SageMaker + EC2) with Streamlit as the frontend.

## Project Structure

- `1A/` — Initial exploration notebook (EDA & model experiments)
- `1B/` — Structured local training pipeline, using MLflow for experiment tracking
- `1C/` — Local app: FastAPI backend + Streamlit frontend
- `2/2A/` — Structured training pipeline (cloud deployment version)
- `2/2B/` — AWS deployment: SageMaker endpoint + FastAPI + Streamlit on EC2

## Workflow

1. **Exploration (1A)** — Initial notebook experiments to understand the data and try out different models.
2. **Training Pipeline (1B)** — A modular pipeline (data loader, preprocessor, trainer, evaluator) that trains several models (Random Forest, XGBoost, LightGBM, MLP) plus their tuned versions, then selects the best one based on macro F1. All experiments are tracked via MLflow.
3. **Local App (1C)** — The best model is served through FastAPI (`backend.py` + `inference.py`), consumed by Streamlit (`frontend.py`) as the input/prediction-result interface.
4. **Cloud Deployment (2A & 2B)** — The model is retrained using the same pipeline, then deployed as an AWS SageMaker endpoint. FastAPI (`api.py`) forwards requests to the SageMaker endpoint, while Streamlit (`frontend.py`) serves as the UI. All services run automatically on EC2 via `user-data.sh` (systemd services for FastAPI on port 8000 and Streamlit on port 8501).

## Model & Results

Best model: **LightGBM (Tuned)**
- Macro F1: 0.7228
- Test Accuracy: 0.7404

Models tried: Random Forest, XGBoost, LightGBM, MLP (each in default and tuned versions via randomized search).

## Tech Stack

- Python
- Scikit-learn, XGBoost, LightGBM
- MLflow (experiment tracking)
- FastAPI (model serving)
- Streamlit (frontend)
- AWS SageMaker & EC2 (cloud deployment)
- boto3

## How to Run Locally (folder 1C)

1. Clone this repository:

   git clone https://github.com/Adhikaxx88/Streamlit-and-AWS-web-deployment.git

   cd Streamlit-and-AWS-web-deployment/1C

2. Install dependencies:

   pip install -r requirements.txt

3. Run the backend (FastAPI):

   uvicorn backend:app --host 0.0.0.0 --port 8000

4. Run the frontend (Streamlit) in a separate terminal:

   streamlit run frontend.py

5. Open your browser at `http://localhost:8501`

## How to Run the Training Pipeline (folder 1B)

1. Go to the `1B/` folder
2. Install dependencies:

   pip install -r requirements.txt

3. Run the pipeline:

   python pipeline.py

The output — model (`.pkl`), label encoder, and evaluation report — will be saved in the `outputs/` folder.

## Deploying to AWS (folder 2/2B)

Cloud deployment happens in two stages:

1. **SageMaker Endpoint** — `deploy_endpoint.py` deploys the model (`inference.py` in the `src/` folder) as a SageMaker endpoint, including a smoke test with sample predictions (Good/Standard/Poor).
2. **EC2 (FastAPI + Streamlit)** — `user-data.sh` runs as a bootstrap script when the EC2 instance is created. It automatically clones the repo, installs dependencies, then runs FastAPI (`api.py`) and Streamlit (`frontend.py`) as systemd services, so they restart automatically on instance reboot.

Note: configuration values such as the S3 bucket name, region, and endpoint name in `deploy_endpoint.py` and `user-data.sh` need to be adjusted to your own AWS environment.

## License

No license specified yet.
