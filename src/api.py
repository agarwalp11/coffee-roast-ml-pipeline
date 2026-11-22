
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Locate the project and saved model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "coffee_roast_model.joblib"

# Load the trained model once when the API starts
model = joblib.load(MODEL_PATH)

# Create the FastAPI application
app = FastAPI(
    title="Coffee Roast Preference Predictor",
    description="Predict preferred coffee roast using survey responses.",
    version="1.0.0"
)


# Define the JSON input structure
class CoffeePreferences(BaseModel):
    brew: str = Field(min_length=1)
    strength: str = Field(min_length=1)
    style: str = Field(min_length=1)
    cups: str = Field(min_length=1)


# Health check endpoint
@app.get("/")
def home():
    return {"message": "Coffee Roast Prediction API is running"}


# Prediction endpoint
@app.post("/predict")
def predict(preferences: CoffeePreferences):

    # Convert JSON input into a DataFrame
    input_data = pd.DataFrame([preferences.model_dump()])

    # Generate a prediction
    prediction = model.predict(input_data)[0]

    # Return prediction as JSON
    return {"predicted_roast": str(prediction)}
