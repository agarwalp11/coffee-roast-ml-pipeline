
from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from ingest import ingest_data
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# MLFlow
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from mlflow.models import infer_signature

#saving model
import joblib



## ==== File Order ======
# Import -- Load config -- Load data -- Split data -- Build model -- Train model
# building mode, training model on training data, and evaluatind model


# Locate the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config():
    """Load pipeline settings from config.yaml."""
    config_path = PROJECT_ROOT / "config.yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def load_data():
    """Download the cleaned dataset from Google Cloud Storage."""
    data_path = ingest_data()

    df = pd.read_csv(data_path)

    return df


# Split the data into training, validation, and test sets
def split_data(df, config):
    """Split data into stratified training, validation, and test sets."""

    feature_columns = config["features"]["categorical"]
    target_column = config["features"]["target"]

    X = df[feature_columns]
    y = df[target_column]

    train_size = config["split"]["train_size"]
    val_size = config["split"]["val_size"]
    test_size = config["split"]["test_size"]
    random_state = config["split"]["random_state"]

    # First split: training data vs. remaining data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=random_state,
        stratify=y
    )

    # Second split: divide remaining data into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_size / (val_size + test_size),
        random_state=random_state,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test



# Build the machine learning model
def build_model(config):
    """Create a pipeline for encoding features and classification."""

    model = Pipeline([
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=config["model"]["max_iter"],
                random_state=config["model"]["random_state"]
            )
        )
    ])

    return model



# Evaluate the model on validation data
def evaluate_model(model, X, y, dataset_name="Validation"):
    """Evaluate the trained model."""

    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")

    print(f"\n{dataset_name} Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro F1-score: {f1:.3f}")

    print("\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred, labels=model.classes_))
    print("Class order:", model.classes_)

    return accuracy, f1



if __name__ == "__main__":
    config = load_config()
    # Store MLflow experiments locally in the project
    mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())

    # Create or select an experiment
    mlflow.set_experiment("coffee-roast-prediction")

    df = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, config
    )

    print("\nDataset splits:")
    print(f"Training: {len(X_train)} rows")
    print(f"Validation: {len(X_val)} rows")
    print(f"Test: {len(X_test)} rows")

    print("\nTraining class distribution:")
    print(y_train.value_counts(normalize=True).round(3))


    # Start a tracked MLflow experiment run
    with mlflow.start_run(run_name="logistic-regression-baseline"):

        # Record model parameters
        mlflow.log_param("model_type", config["model"]["type"])
        mlflow.log_param("max_iter", config["model"]["max_iter"])
        mlflow.log_param("random_state", config["model"]["random_state"])
        mlflow.log_param("train_size", config["split"]["train_size"])
        mlflow.log_param("val_size", config["split"]["val_size"])
        mlflow.log_param("test_size", config["split"]["test_size"])

        # Build and train the model
        model = build_model(config)
        model.fit(X_train, y_train)

        print("\nModel trained successfully!")
        print("Model classes:", model.classes_)

        # Evaluate the model
        accuracy, f1 = evaluate_model(model, X_val, y_val, dataset_name="Validation")

        # Record validation metrics
        mlflow.log_metric("validation_accuracy", accuracy)
        mlflow.log_metric("validation_macro_f1", f1)

        # Final evaluation using untouched test data
        print("\n--- Final Test Evaluation ---")

        test_accuracy, test_f1 = evaluate_model(model, X_test, y_test, dataset_name="Test")

        # Record final test results in MLflow
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_macro_f1", test_f1)


        # Save trained model for API deployment
        # reads the save location from config.yaml
        model_path = PROJECT_ROOT / config["output"]["model_path"]

        # Create the models directory if needed
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the complete preprocessing and classification pipeline
        joblib.dump(model, model_path)

        print(f"\nModel saved successfully to: {model_path}")


        # Describe the model's expected inputs and outputs
        input_example = X_train.head(5)
        signature = infer_signature(
            input_example,
            model.predict(input_example)
        )

        # Log the trained model with its signature
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        # Create a confusion matrix plot
        fig, ax = plt.subplots(figsize=(7, 5))

        ConfusionMatrixDisplay.from_estimator(
            model,
            X_val,
            y_val,
            labels=model.classes_,
            cmap="Blues",
            ax=ax
        )

        ax.set_title("Coffee Roast Preference - Validation Confusion Matrix")

        # Log the plot as an MLflow artifact
        mlflow.log_figure(fig, "plots/confusion_matrix.png")
        plt.close(fig)

        print("\nModel and confusion matrix logged successfully!")


        print("\nExperiment logged successfully to MLflow!")


