import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Page title
st.title("LG Pricing Optimization Model")
st.markdown("Optimize product pricing based on demand elasticity, competitor pricing, and customer preferences.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("LG_optimized_pricing_data.csv")

data = load_data()

# Display dataset
if st.checkbox("Show Dataset"):
    st.write(data.head())

# Clean data
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# Visualizations
st.subheader("Visualizations")

# Bar plot of feature impact (dummy until model is trained)
st.write("### Correlation Heatmap")
fig1, ax1 = plt.subplots()
sns.heatmap(data[['Discounted_Price', 'Competitor_Price', 'Purchased_Quantity']].corr(), annot=True, cmap="coolwarm", ax=ax1)
st.pyplot(fig1)

st.write("### Demand Distribution by Product Type")
fig2, ax2 = plt.subplots()
sns.boxplot(data=data, x='Product_Type', y='Purchased_Quantity', palette='Set2', ax=ax2)
st.pyplot(fig2)

# Prepare data for model
X = data[['Discounted_Price', 'Competitor_Price', 'Product_Type']]
y = data['Purchased_Quantity']

# Preprocessing and pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ['Product_Type'])
    ],
    remainder='passthrough'
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
model_pipeline.fit(X, y)

# Save model
joblib.dump(model_pipeline, 'pricing_demand_model.pkl')

# Predict section
st.subheader("Predict Purchased Quantity")
product_type = st.selectbox("Select Product Type", data['Product_Type'].unique())
discounted_price = st.number_input("Discounted Price", min_value=0.0)
competitor_price = st.number_input("Competitor Price", min_value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame({
        'Discounted_Price': [discounted_price],
        'Competitor_Price': [competitor_price],
        'Product_Type': [product_type]
    })
    prediction = model_pipeline.predict(input_df)[0]
    st.success(f"Predicted Purchased Quantity: {prediction:.2f}")
