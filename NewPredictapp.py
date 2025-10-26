
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "joblib"], check=False)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor

st.set_page_config(page_title="Accra Rental Price Predictor", page_icon="🏠", layout="wide")

# -------------------------------------------
# Load model and data
# -------------------------------------------

try:
    model = joblib.load("catboost_model.pkl")
    df = pd.read_excel("RESIDENTIAL DATA.xlsx")
except FileNotFoundError:
    st.error("❌ Model or data file not found. Please ensure both files are in the same folder.")
    st.stop()

# -------------------------------------------
# Prepare locality information
# -------------------------------------------
df['LOCALITY'] = df['LOCALITY'].str.strip().str.title()
localities = sorted(df['LOCALITY'].unique().tolist())

# Mapping for unseen localities
locality_mapping = {
    "East Airport": "Airport Residential Area",
    "New Weija Estate": "Weija",
    "Community 25": "Tema",
    "Tema Community 25": "Tema",
    "North Ridge": "Ridge",
    "West Ridge": "Ridge",
    "Ashalaja": "Amasaman",
    "Haatso-Westlands": "Haatso",
    "Adenta West": "Adenta",
    "Oyarifa Hills": "Oyarifa",
}

# -------------------------------------------
# Streamlit layout
# -------------------------------------------
st.title("🏘️ Residential Rental Price Predictor - Greater Accra Region")

st.markdown(
    "This app predicts **rental prices** for properties within the **Greater Accra Region** using a trained CatBoost machine learning model.  \n"
    "Select property details and a locality to get an estimated monthly rent."
)

# Sidebar for user input
st.sidebar.header("🏠 Property Details")

property_type = st.sidebar.selectbox("Property Type", ["Apartment", "House", "Duplex", "Townhouse"])
bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=1, max_value=10, step=1)
floor_area = st.sidebar.number_input("Floor Area (m²)", min_value=10, max_value=5000, step=10)
condition = st.sidebar.selectbox("Condition", ["New", "Used"])
furnishing = st.sidebar.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])

# Locality input (dropdown + manual option)
st.sidebar.subheader("📍 Location")
locality_input = st.sidebar.text_input(
    "Enter Locality Name (e.g., East Legon, Spintex, Madina):",
    placeholder="Type or choose below..."
)
selected_locality = st.sidebar.selectbox("Or select from known areas:", [""] + localities)

# Determine final locality
if locality_input:
    locality = locality_input.strip().title()
else:
    locality = selected_locality.strip().title()

# Handle unseen locality
mapped_note = ""
if locality not in localities:
    mapped_locality = locality_mapping.get(locality, None)
    if mapped_locality:
        st.warning(f"⚠️ '{locality}' not found in dataset. Using similar area '{mapped_locality}' for prediction.")
        locality = mapped_locality
        mapped_note = f"(Mapped from {locality_input})"
    else:
        locality = "Accra"  # fallback to general city mean

# -------------------------------------------
# Prepare input data
# -------------------------------------------
property_df = pd.DataFrame({
    "BEDROOMS": [bedrooms],
    "BATHROOMS": [bathrooms],
    "FLOOR AREA": [floor_area],
    "CONDITION": [condition],
    "FURNISHING STATUS": [furnishing],
    "LOCALITY": [locality],
    "PROPERTY TYPE": [property_type],
})

# Encode features as done during training
df_copy = df.copy()
combined = pd.concat([df_copy, property_df], ignore_index=True)

# One-hot encode
encoded = pd.get_dummies(combined, columns=["PROPERTY TYPE", "CONDITION", "FURNISHING STATUS"], drop_first=False)
encoded = pd.get_dummies(encoded, columns=["LOCALITY"], drop_first=False)

# Match model columns
X = encoded.tail(1)
model_features = model.feature_names_ if hasattr(model, "feature_names_") else X.columns
for col in model_features:
    if col not in X.columns:
        X[col] = 0
X = X[model_features]

# -------------------------------------------
# Predict rental price
# -------------------------------------------
if st.sidebar.button("🔍 Predict Rental Price"):
    try:
        log_price = model.predict(X)[0]
        predicted_price = round(float(np.exp(log_price)), 2)

        st.success(f"💰 Estimated Monthly Rent: **₵{predicted_price:,.2f}** {mapped_note}")
        st.balloons()

        # Show summary
        st.subheader("Property Summary")
        st.write(property_df)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------------------------------
# Display dataset locality summary
# -------------------------------------------
with st.expander("📊 View Average Prices by Locality"):
    avg_prices = df.groupby("LOCALITY")["PRICE"].mean().reset_index().sort_values("PRICE", ascending=False)
    st.dataframe(avg_prices.style.format({"PRICE": "₵{:.2f}"}))

st.markdown("---")
st.caption("Developed by Tracy Abena Benneh | Machine Learning Project (2025)")
