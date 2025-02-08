# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer

app = FastAPI(
    title="Solar Energy Forecast API",
    description="Forecast solar energy and efficiency for the next 7 days given location and panel area.",
    version="1.0.0",
)

# Enable CORS (optional, in case you want to call the API from a web app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model (ensure solar_model.pkl is in your repository root)
try:
    model = joblib.load("solar_model.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")


def get_api_data(latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Call the Open-Meteo forecast API for the given period.
    Returns a DataFrame with hourly data, including temperature_2m, ghi, and cloud_cover.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,shortwave_radiation,cloud_cover",
        "timezone": "auto",
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data from API: {e}")
    
    if 'hourly' not in data or not data['hourly'].get('time'):
        raise HTTPException(status_code=404, detail="No hourly data available from API for the specified period.")
    
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(data['hourly']['time']),
        'temperature_2m': data['hourly'].get('temperature_2m'),
        'ghi': data['hourly'].get('shortwave_radiation'),
        'cloud_cover': data['hourly'].get('cloud_cover'),
    })
    
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    return df

def prepare_weighted_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate the 'ghi' and 'cloud_cover' columns 4 times each to match the training format.
    """
    required_features = ["ghi", "hour", "month", "cloud_cover"]
    for col in required_features:
        if col not in df.columns:
            df[col] = np.nan

    X_orig = df[required_features]
    X_weighted = pd.concat([
        X_orig[['hour']],       # 1 copy of hour
        X_orig[['month']],      # 1 copy of month
        X_orig[['ghi']],        # replicate ghi 4 times
        X_orig[['ghi']],
        X_orig[['ghi']],
        X_orig[['ghi']],
        X_orig[['cloud_cover']],# replicate cloud_cover 4 times
        X_orig[['cloud_cover']],
        X_orig[['cloud_cover']],
        X_orig[['cloud_cover']]
    ], axis=1)
    
    X_weighted.columns = [
        "hour", "month",
        "ghi_1", "ghi_2", "ghi_3", "ghi_4",
        "cloud_cover_1", "cloud_cover_2", "cloud_cover_3", "cloud_cover_4"
    ]
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X_weighted), columns=X_weighted.columns)
    return X_imputed

@app.get("/forecast")
def forecast(
    latitude: float,
    longitude: float,
    start_date: str,  # Expected format: YYYY-MM-DD
    panel_area: float
):
    """
    Forecast solar energy and panel efficiency for 7 days starting from start_date.
    Returns a list of hourly forecasts with timestamps, predicted energy, computed energy, and efficiency.
    """
    try:
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    
    end_date = (start_date_obj + timedelta(days=6)).strftime("%Y-%m-%d")
    
    # Fetch forecast data
    df = get_api_data(latitude, longitude, start_date, end_date)
    if df.empty:
        raise HTTPException(status_code=404, detail="No forecast data available for the specified period.")
    
    # Prepare features for model prediction
    X = prepare_weighted_features(df)
    predictions = model.predict(X)
    df["predicted_energy_Wh"] = predictions

    # Compute physical model values:
    BASE_EFF = 0.15
    TEMP_COEFF = -0.004
    TEMP_REF = 25

    # Adjust GHI based on cloud cover (assuming cloud_cover is in %)
    df["ghi_adjusted"] = df["ghi"] * (1 - df["cloud_cover"] / 100.0)
    df["panel_temp"] = df["temperature_2m"] + df["ghi_adjusted"] / 800.0
    df["efficiency"] = BASE_EFF * (1 + TEMP_COEFF * (df["panel_temp"] - TEMP_REF))
    df["computed_energy_Wh"] = panel_area * df["efficiency"] * df["ghi_adjusted"]

    # Prepare JSON response
    result_df = df[["timestamp", "predicted_energy_Wh", "computed_energy_Wh", "efficiency"]].copy()
    # Convert timestamp to string for JSON serialization
    result_df["timestamp"] = result_df["timestamp"].astype(str)
    return result_df.to_dict(orient="records")

# Run with: uvicorn main:app --host 0.0.0.0 --port $PORT
