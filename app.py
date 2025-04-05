from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load data & model
data = pd.read_csv("LG_optimized_pricing_data.csv")
model = joblib.load("pricing_demand_model.pkl")

# Create FastAPI app
app = FastAPI()

# Define request body model
class PredictionRequest(BaseModel):
    Discounted_Price: float
    Competitor_Price: float
    Product_Type: str

@app.get("/")
def home():
    return {"message": "LG Pricing Optimization API is live"}

@app.post("/predict")
def predict(req: PredictionRequest):
    input_df = pd.DataFrame({
        'Discounted_Price': [req.Discounted_Price],
        'Competitor_Price': [req.Competitor_Price],
        'Product_Type': [req.Product_Type]
    })
    prediction = model.predict(input_df)[0]
    return {"predicted_purchased_quantity": prediction}
