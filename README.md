# ☕ Coffee Roast Preference Predictor

An end-to-end machine learning pipeline that predicts a coffee drinker's preferred roast level (**Light**, **Medium**, or **Dark**) from their brewing habits, preferred strength, flavor characteristics, and daily coffee consumption.

**[Try the live Streamlit app](https://coffee-roast-predictor-ml-pipeline.streamlit.app/)** · **[Deployed FastAPI documentation](https://coffee-roast-api-70362195490.us-east1.run.app/docs)** · **[GitHub repository](https://github.com/agarwalp11/coffee-roast-ml-pipeline)**

## Project overview and goals

This project demonstrates a reproducible, cloud-deployed machine learning workflow: preprocess survey data, store the processed dataset in cloud storage, train and evaluate a classification model, track experiments, package the workflow with Docker, serve predictions through an API, and provide a publicly accessible frontend.

The project was developed for **Duke University's AIPI 510 — Data Science Foundations, Module Project 3: Data/ML Engineering**.

## Live applications

| Component | Link | Purpose |
| --- | --- | --- |
| Streamlit frontend | [Coffee Roast Predictor](https://coffee-roast-predictor-ml-pipeline.streamlit.app/) | Collects preferences and displays a prediction |
| Cloud Run API | [API base URL](https://coffee-roast-api-70362195490.us-east1.run.app/) | Hosts the trained model |
| Interactive API docs | [FastAPI Swagger UI](https://coffee-roast-api-70362195490.us-east1.run.app/docs) | Tests the prediction endpoint |

### System architecture

```text
Great American Coffee Taste Test survey (TidyTuesday)
                       |
                       v
             Data preprocessing
                       |
                       v
             Google Cloud Storage
                       |
                       v
          Config-driven model training
         (scikit-learn + MLflow)
                       |
                       v
             Saved model artifact
                       |
                       v
          Dockerized FastAPI service
             (Google Cloud Run)
                       ^
                       | HTTP POST /predict
                       |
            Streamlit Community Cloud
                       ^
                       |
                      User
```
The data from Great American Coffee Taste Test survey (TidyTuesday), Preproccsed, cleaned handling missing values and uploaded to Google CLoud Storage. The model was then trainined to predict whether someone prefers light, medium, or dark roast. Experiments were tracked, evaluatings models performance. Once it was trained and evaluated it was then saved so it can be reused without having to retrain it. A docker was created to package the app, dependencies, and files and then deployed through FastAPI, which allows other applications to request predictions. Streamlist front end app is created so users can enter their coffee preferences. 

The frontend sends requests to the **deployed API** (FastAPI), rather than loading the model directly in Streamlit. API processes inputs, runs saved model, and then returns a prediction to Streamlit App

## Dataset

**Source:** Great American Coffee Taste Test survey, published through the [TidyTuesday project](https://github.com/rfordatascience/tidytuesday/tree/master/data/2024/2024-05-14).

The raw survey contains coffee preferences and tasting responses. For this project  four categorical predictors were selected that could determine a users roast preference and one classification target, the roast preference:

| Column | Description | Example |
| --- | --- | --- |
| `brew` | Brewing method(s) | `Pour over`, `Espresso`, `French press` |
| `strength` | Preferred coffee strength | `Somewhat light`, `Medium`, `Somewhat strong` |
| `style` | Preferred flavor characteristic | `Fruity`, `Chocolatey`, `Nutty` |
| `cups` | Cups consumed per day | `Less than 1`, `1`, `2`, `3` |
| `roast_level` | **Target:** preferred roast level | `Light`, `Medium`, `Dark` |

The processed dataset contains **3,940 observations**. Class distribution: **Light 1,930**, **Medium 1,557**, **Dark 453**. The processed CSV is stored in a private Google Cloud Storage bucket and is downloaded by the ingestion script for model training.

### Preprocessing

`src/preprocess.py`:

1. Loads the raw survey CSV.
2. Selects the four predictors and the roast target.
3. Removes records without a usable target.
4. Replaces missing predictor values with `Unknown`.
5. Maps **Blonde/Nordic → Light** and **French/Italian → Dark**.
6. Writes `data/processed/coffee_cleaned.csv`.

Run from the repository root:

```bash
python -m src.preprocess
```

> **Note:** The survey allowed multiple brewing methods, and the model treats each full combination string as a categorical value. The frontend currently offers individual brewing methods to keep input simple.

## Model architecture and evaluation

The model is a scikit-learn **Pipeline** combining:

- **OneHotEncoder** for the four categorical features, with unknown categories ignored.
- **LogisticRegression** for multiclass roast-level prediction.

Data is split with stratification into **70% training (2,758 rows)**, **15% validation (591 rows)**, and **15% held-out test (591 rows)**. The preprocessing encoder is fitted on the training data within the pipeline to avoid leakage. Configuration is maintained in `config.yaml`.

### Results

| Metric | Validation | Held-out test |
| --- | ---: | ---: |
| Accuracy | 0.721 | **0.734** |
| Macro F1 | 0.637 | **0.645** |

A majority-class classifier would achieve approximately **49% accuracy** if there isn't model training. The model exceeds this baseline, but performance is weaker for the less common **Dark** class (test recall approximately **0.32**). Accuracy alone should not be interpreted as equal performance across roast categories. Next step would be to understand if we need more predictors or if any of the current predictors establishes noise

### Experiment tracking

Training uses **MLflow** to log model parameters, validation/test metrics, the fitted model, and a confusion-matrix artifact. Run MLflow locally after training:

```bash
mlflow ui
```

Open <http://localhost:5000> to inspect recorded experiments. Local `mlruns/` data is not necessarily committed to the repository; rerun training to generate your own experiment records.

## Technologies and cloud services

| Technology | Role |
| --- | --- |
| Python, pandas | Data preparation |
| scikit-learn | Feature encoding, model training, evaluation |
| MLflow | Experiment tracking |
| Google Cloud Storage | Hosting the processed dataset |
| FastAPI, Uvicorn | Prediction API |
| Docker | Reproducible container environment |
| Google Cloud Run | Public API deployment |
| Streamlit Community Cloud | Public frontend hosting |
| GitHub | Version control, branches, and pull requests |

## Setup and usage

### 1. Clone and install

Use Python 3.11 (the Docker image uses Python 3.11):

```bash
git clone https://github.com/agarwalp11/coffee-roast-ml-pipeline.git
cd coffee-roast-ml-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.

### 2. Prepare or download data

To regenerate the cleaned CSV from the raw survey (if the raw CSV is available at the path expected by `src/preprocess.py`):

```bash
python -m src.preprocess
```

To download the hosted processed data from Google Cloud Storage:

```bash
gcloud auth application-default login
python -m src.ingest
```

**Cloud access:** The GCS bucket is private. Downloading requires an authorized Google Cloud identity and the bucket configuration used by `src/ingest.py`. Users without access can use the public source dataset and the preprocessing script instead.

### 3. Train and evaluate

```bash
python -m src.train
```

Training reads the configuration in `config.yaml`, evaluates the model, logs the run to MLflow, and saves the fitted pipeline to:

```text
models/coffee_roast_model.joblib
```

### 4. Run the API locally

```bash
python -m uvicorn src.api:app --reload
```

Open <http://localhost:8000/docs> and use **POST `/predict`** to test an input. Example:

```json
{
  "brew": "French press",
  "strength": "Somewhat light",
  "style": "Chocolatey",
  "cups": "3"
}
```

The API returns a JSON object in this format (the actual class depends on the model):

```json
{
  "predicted_roast": "Medium"
}
```

### 5. Build and run with Docker

From the repository root:

```bash
docker build -t coffee-roast-ml .
```

To serve the API in a container:

```bash
docker run --rm -p 8080:8080 coffee-roast-ml
```

Open <http://localhost:8080/docs>. The Dockerfile starts the API on port `8080` by default, or on the port provided through the `PORT` environment variable.

To run the training entry point in the image instead of the API, override the image command:

```bash
docker run --rm coffee-roast-ml python -m src.train
```

**Note:** Training that depends on the private GCS bucket requires appropriate credentials inside the container. Model artifacts created in an ephemeral container are not persisted to your host unless you mount a volume.

### 6. Deploy the API to Google Cloud Run

Install and authenticate the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), select a project, and enable the required services:

```bash
gcloud auth login
gcloud config set project YOUR_GOOGLE_CLOUD_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

Deploy from the repository root:

```bash
gcloud run deploy coffee-roast-api \
  --source . \
  --region us-east1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --max-instances 2
```

Cloud Run builds and deploys the container. The deployed endpoint is:

- API: <https://coffee-roast-api-70362195490.us-east1.run.app/>
- Docs: <https://coffee-roast-api-70362195490.us-east1.run.app/docs>
- Prediction endpoint: `POST https://coffee-roast-api-70362195490.us-east1.run.app/predict`

### 7. Run the Streamlit frontend locally

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

Open <http://localhost:8501>. The frontend posts the selected preferences to the deployed Cloud Run API and displays its predicted roast level.

### 8. Deploy the frontend

1. Push `app/streamlit_app.py` and `app/requirements.txt` to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), deploy from the GitHub repository's `main` branch.
3. Set the entry-point file to `app/streamlit_app.py`.
4. Confirm the frontend successfully calls the Cloud Run API.

**Live frontend:** <https://coffee-roast-predictor-ml-pipeline.streamlit.app/>

## Repository structure

```text
coffee-roast-ml-pipeline/
├── app/
│   ├── streamlit_app.py       # Public frontend
│   └── requirements.txt       # Frontend dependencies
├── data/
│   └── processed/            # Cleaned/downloaded data (local)
├── models/
│   └── coffee_roast_model.joblib
├── src/
│   ├── preprocess.py         # Survey cleaning
│   ├── ingest.py             # GCS download
│   ├── train.py              # Training, evaluation, MLflow
│   └── api.py                # FastAPI inference service
├── config.yaml               # Pipeline configuration
├── Dockerfile                # Containerized training and API
├── .dockerignore
├── requirements.txt          # Pipeline/API dependencies
└── README.md
```

## Reproducibility and development practices
- Centralized model/pipeline settings in `config.yaml` for model settings, training configuration, and making it easier to rerun experiments consistently
- Data stored in Google Cloud Storage for centralized location for accessing the data 
- Stratified training, validation, and test partitions to maintain similar roast-level class distribution across datasets
- Feature encoding inside the fitted scikit-learn pipeline.
- MLflow logging for experiment comparison.
- Docker for consistent runtime packaging.
- Saved trained model and preprocessing pipeline so predictions can be generated without training the model 
- Separate API and frontend deployments.
- Scripts for API, data ingestion, inspecting data, prepprocessing, and trainiong 
- Github used to track changes, Git feature branches and pull requests for incremental development.

## Limitations and future improvements

- **Class imbalance:** Dark roast is less common and has lower recall. Future work could evaluate class weighting, resampling, and additional metrics.
- **Brewing-method combinations:** The survey includes multi-select responses stored as complete strings. A future version could encode individual brewing methods as separate binary features and retrain the model.
- **Unknown categories:** Inputs not observed in training are ignored by the encoder; matching the frontend options to training categories reduces this issue.
- **Survey-based predictions:** Results reflect patterns in a specific survey, not objective recommendations or guaranteed personal preferences. This survey was collected by a what most likley would be from coffee enthusiasts 
- **Operational improvements:** Add automated tests, continuous integration, and deployment configuration for a more production-ready pipeline.

## Ethical Considerations and Limitations

Although this is a low-risk application, there are some important considerations:

- **Data Privacy:** The model uses public survey data and does not require personal information.
- **Dataset Bias:** Survey participants may not represent all coffee drinkers, which could affect prediction accuracy.
- **Class Imbalance:** Dark roast is underrepresented in the dataset, making predictions for this category less reliable.
- **Model Limitations:** The model uses only four features, so predictions should be treated as suggestions rather than definitive results.

