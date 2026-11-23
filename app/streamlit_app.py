
import streamlit as st
import requests

# Deployed FastAPI endpoint on Google Cloud Run
API_URL = "https://coffee-roast-api-70362195490.us-east1.run.app/predict"

st.set_page_config(
    page_title="Coffee Roast Predictor",
    page_icon="☕",
    layout="centered"
)

# Custom coffee-themed styling
st.markdown("""
<style>
    .stApp {
        background-color: #F6F0E7;
        color: #38271F;
    }

    .block-container {
        max-width: 760px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #38271F, #694734);
        padding: 35px 25px;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: #FFF5E9;
        font-size: 2.3rem;
        margin-bottom: 10px;
    }

    .hero p {
        color: #E8C9A2;
        font-size: 1.05rem;
        margin: 0;
    }

    .stButton > button {
        background-color: #6B4532;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 20px;
        font-weight: 600;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #38271F;
        color: white;
        border: none;
    }

    .result-card {
        background-color: #EDE1D1;
        border: 1px solid #D7C1A9;
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        margin-top: 20px;
    }

    .result-card h2 {
        color: #38271F;
        font-size: 2rem;
        margin: 10px 0;
    }

    .result-card p {
        color: #6B4532;
    }

    div[data-testid="stWidgetLabel"] p {
        color: #38271F;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero">
    <h1>☕ Find Your Perfect Roast</h1>
    <p>Discover the coffee roast that matches your taste.</p>
</div>
""", unsafe_allow_html=True)

st.subheader("Your Coffee Preferences")
st.caption("Answer four quick questions to discover your roast.")

# Collect user preferences
with st.form("coffee_preferences"):

    brew = st.selectbox(
        "☕ How do you brew your coffee?",
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

    col1, col2 = st.columns(2)

    with col1:
        strength = st.selectbox(
            "Coffee strength",
            [
                "Weak",
                "Somewhat light",
                "Medium",
                "Somewhat strong",
                "Very strong"
            ]
        )

    with col2:
        style = st.selectbox(
            "Preferred flavor",
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
        "How many cups do you drink per day?",
        [
            "Less than 1",
            "1",
            "2",
            "3",
            "4",
            "More than 4"
        ]
    )

    st.write("")

    submitted = st.form_submit_button(
        "✨ Predict My Perfect Roast",
        use_container_width=True
    )

# Send preferences to deployed API
if submitted:

    payload = {
        "brew": brew,
        "strength": strength,
        "style": style,
        "cups": cups
    }

    try:
        with st.spinner("Finding your perfect roast..."):
            response = requests.post(
                API_URL,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            prediction = response.json()["predicted_roast"]

        # Display prediction in a styled card
        st.markdown(f"""
        <div class="result-card">
            <p>YOUR ROAST RECOMMENDATION</p>
            <h2>☕ {prediction} Roast</h2>
            <p>Based on your coffee preferences</p>
        </div>
        """, unsafe_allow_html=True)


    except requests.exceptions.RequestException as error:
        st.error(
            f"Unable to connect to the prediction API: {error}"
        )

    except (KeyError, ValueError):
        st.error("The API returned an unexpected response.")

# Footer
st.divider()
st.caption(
    "Built with Streamlit, FastAPI, Google Cloud Run & Machine Learning"
)
