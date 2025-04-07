import streamlit as st
import pandas as pd
import numpy as np
import joblib

#Title
st.title("📈 Maryam's LG Pricing Optimization Model")
st.markdown("Predict purchased quantity based on price, product type, and competitor pricing.")

#Load model
@st.cache_resource
def load_model():
    return joblib.load("pricing_demand_model.pkl")

model = load_model()

# Load dataset for form options
@st.cache_data
def load_data():
    return pd.read_csv("LG_optimized_pricing_data.csv")

data = load_data()

# Input form
st.subheader("🔢 Input Features")

product_type = st.selectbox("Product Type", options=data['Product_Type'].unique())
discounted_price = st.number_input("Discounted Price", min_value=0.0)
competitor_price = st.number_input("Competitor Price", min_value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame({
        "Discounted_Price": [discounted_price],
        "Competitor_Price": [competitor_price],
        "Product_Type": [product_type]
    })
    prediction = model.predict(input_df)[0]
    st.success(f"✅ Predicted Purchased Quantity: {prediction:.2f}")
