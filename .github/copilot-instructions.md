# ChurnWatch AI Agent Instructions

## Project Overview
**ChurnWatch** is a real-time churn prediction and dashboard platform. The project uses ML models (XGBoost, CatBoost, LightGBM) with MLflow tracking and serves predictions via FastAPI with Streamlit dashboards.

## Architecture & Data Flow
- **Data Source**: `data-csv/costumer-churn.csv` - customer churn dataset for model training
- **ML Stack**: scikit-learn, imbalanced-learn (class imbalance handling), XGBoost/CatBoost/LightGBM ensembles
- **Experiment Tracking**: MLflow for tracking model versions and metrics
- **Backend**: FastAPI (Uvicorn) for prediction API
- **Frontend**: Streamlit for interactive dashboards
- **Workflows**: Prefect for orchestrating ETL and training pipelines
- **Database**: PostgreSQL (sqlalchemy ORM)

## Project Structure
```
app.py                 # FastAPI application (main entry point)
setup.py              # Package config; project name "churnrader"
requirements.txt      # ML, API, orchestration, logging dependencies
data-csv/
  ├── costumer-churn.csv
lab/
  └── trials.ipynb    # Experimentation notebook
```

## Key Patterns & Conventions

### 1. Model Training Workflow
- Use `imbalanced-learn` for handling class imbalance in churn data
- Track all model runs with MLflow (metrics, parameters, artifacts)
- Ensemble approach: compare XGBoost, CatBoost, LightGBM, select best performer
- Validation: cross-validation before production deployment

### 2. API Development
- FastAPI for REST endpoints (see `app.py`)
- Prediction endpoint structure: accept customer features → return churn probability
- Use SQLAlchemy ORM for database interactions

### 3. Dashboard & Visualization
- Streamlit for interactive UI; Plotly for charts
- Connect to PostgreSQL for live data queries
- Prefect orchestrates dashboard data refresh jobs

### 4. Configuration & Logging
- Use `python-dotenv` for environment variables (database URLs, API keys)
- `loguru` for structured logging across modules
- Environment-specific settings: development vs. production modes

### 5. Experimentation
- Use `lab/trials.ipynb` for exploratory analysis and model experiments
- Prototype new features in notebook before integrating into app.py
- Log all trial results to MLflow for comparison

## Critical Commands
```bash
# Setup
pip install -r requirements.txt
python setup.py develop

# Run API
uvicorn app:app --reload

# Run Streamlit Dashboard
streamlit run app.py

# MLflow UI
mlflow ui

# Train model (when workflow is implemented)
python -m prefect.cli.server start  # Start Prefect server
# Define flows in code and trigger runs
```

## Developer Notes
- **Naming**: Project package is "churnrader" (setup.py) but repo is "ChurnWatch"
- **Data Path**: All data reads reference `data-csv/` relative path
- **Test Environment**: `churnenv` conda environment is active (from terminal context)
- **Database**: PostgreSQL required; configure via `.env` file with connection string
