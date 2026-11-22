
import streamlit as st
import requests

# Deployed FastAPI endpoint on Google Cloud Run
API_URL = "https://coffee-roast-api-70362195490.us-east1.run.app/predict"

st.set_page_config(page_title="Coffee Roast Predictor", page_icon="☕")

st.title("☕ Coffee Roast Predictor")
st.write("Discover your preferred coffee roast based on your coffee habits!")

# Collect user preferences
# with drop down menus for each question
brew = st.selectbox(
    "How do you brew your coffee?",
    [
        "Pour over",
        "Espresso",
        "French press",
        "Coffee brewing machine (e.g. Mr. Coffee)",
        "Pod/capsule machine (e.g. Keurig/Nespresso)",
        "Cold brew",
        "Bean-to-cup machine",
        "Instant coffee",
        "Coffee extract (e.g. Cometeer)",
        "Other"
    ]
)

strength = st.selectbox(
    "How strong do you like your coffee?",
    ["Weak", "Somewhat light", "Medium", "Somewhat strong", "Very strong"]
)

style = st.selectbox(
    "What flavor characteristics do you prefer?",
    [
        "Fruity",
        "Chocolatey",
        "Full Bodied",
        "Bright",
        "Nutty",
        "Sweet",
        "Caramalized",
        "Juicy",
        "Bold",
        "Floral"
    ]
)

cups = st.selectbox(
    "How many cups of coffee do you drink per day?",
    ["Less than 1", "1", "2", "3", "4", "More than 4"]
)

# Send preferences to the deployed model API
if st.button("Predict My Roast"):
    payload = {
        "brew": brew,
        "strength": strength,
        "style": style,
        "cups": cups
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()

        prediction = response.json()["predicted_roast"]

        st.success(f"Your predicted coffee roast is: {prediction}!")

    except requests.exceptions.RequestException as error:
        st.error(f"Unable to connect to the prediction API: {error}")
