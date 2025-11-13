from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import keras
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
## added changes here

# Initialize FastAPI app
app = FastAPI(title="Hotel Cancellation Prediction API")

# Serve static files (CSS, JS, etc. if you have any)
app.mount("/static", StaticFiles(directory="."), name="static")

# Define the input data model
class HotelBooking(BaseModel):
    hotel: str
    arrival_date_month: str
    meal: str
    market_segment: str
    distribution_channel: str
    reserved_room_type: str
    deposit_type: str
    customer_type: str
    lead_time: float
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: float  # Can be float due to possible missing values
    babies: int
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    required_car_parking_spaces: int
    total_of_special_requests: int
    adr: float


model = keras.saving.load_model("hotel_cancellation_model_with_dropout.keras") 
    


features_num = [
    "lead_time", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "required_car_parking_spaces",
    "total_of_special_requests", "adr",
]

features_cat = [
    "hotel", "arrival_date_month", "meal",
    "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "customer_type",
]

# Create the same preprocessor used during training
transformer_num = make_pipeline(
    SimpleImputer(strategy="constant"),  # there are a few missing values
    StandardScaler(),
)

transformer_cat = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="NA"),
    OneHotEncoder(handle_unknown='ignore'),
)

preprocessor = make_column_transformer(
    (transformer_num, features_num),
    (transformer_cat, features_cat),
)

# Month mapping dictionary
month_map = {
    'January': 1, 'February': 2, 'March': 3,
    'April': 4, 'May': 5, 'June': 6, 'July': 7,
    'August': 8, 'September': 9, 'October': 10,
    'November': 11, 'December': 12
}

@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    return FileResponse("index.html")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict_cancellation(booking: HotelBooking):
    """Predict hotel cancellation"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check the model file.")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([booking.dict()])
        
        # Apply the same preprocessing as during training
        # Map month name to number
        input_data['arrival_date_month'] = input_data['arrival_date_month'].map(month_map)
        
        # Preprocess the input data
        processed_data = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)
        
        # Return results
        return {
            "prediction": int(prediction[0]),
            "probability_cancelled": float(prediction_proba[0][1]),
            "probability_not_cancelled": float(prediction_proba[0][0])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/features-info")
async def get_features_info():
    """Return information about expected features and their values"""
    return {
        "categorical_features": {
            "hotel": ["Resort Hotel", "City Hotel"],
            "arrival_date_month": ["January", "February", "March", "April", "May", "June", 
                                 "July", "August", "September", "October", "November", "December"],
            "meal": ["BB", "FB", "HB", "SC", "Undefined"],
            "market_segment": ["Direct", "Corporate", "Online TA", "Offline TA/TO", 
                             "Complementary", "Groups", "Undefined", "Aviation"],
            "distribution_channel": ["Direct", "Corporate", "TA/TO", "Undefined", "GDS"],
            "reserved_room_type": ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"],
            "deposit_type": ["No Deposit", "Refundable", "Non Refund"],
            "customer_type": ["Transient", "Contract", "Transient-Party", "Group"]
        },
        "numerical_features": features_num
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)