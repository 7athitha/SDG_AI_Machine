import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import streamlit as st
import requests
import matplotlib.pyplot as plt
from datetime import datetime

# API Configuration
API_KEY = "YOUR_VALID_OPENWEATHERMAP_API_KEY"  # Replace with your actual API key
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# Fetch real-time weather data
def get_real_time_weather(lat, lon):
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric"
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        return {
            "cloudCoverage": data["clouds"]["all"],
            "temperature": data["main"]["temp"],
            "windSpeed": data["wind"]["speed"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    else:
        st.error(f"API Error: {response.status_code} - {response.text}")
        return None

# Load historical data
@st.cache_data
def load_data():
    data = pd.read_csv('weather.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    data['solar_potential'] = 1 - (data['cloudCoverage'] / 100)
    return data

# Prepare features and train models
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    predictions = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        trained_models[name] = model
        predictions[name] = pred
    
    return trained_models, predictions, X_test, y_test, scaler

# Estimate CO2 reduction (simplified example)
def estimate_co2_reduction(solar_potential):
    # Assume 1 unit of solar potential reduces 0.5 kg CO2/day (hypothetical)
    return solar_potential * 0.5

# Streamlit App
st.set_page_config(layout="wide")
st.title("Solar Energy Potential Predictor 🌞")
st.write("Empower sustainable energy with real-time weather data and ML model comparisons. Aligns with SDG 7: Affordable and Clean Energy.")

# Sidebar for navigation
st.sidebar.header("Settings")
location_option = st.sidebar.selectbox("Select Location", ["Custom", "New York", "London", "Tokyo"])
if location_option == "New York":
    lat, lon = 40.7128, -74.0060
elif location_option == "London":
    lat, lon = 51.5074, -0.1278
elif location_option == "Tokyo":
    lat, lon = 35.6762, 139.6503
else:
    lat = st.sidebar.number_input("Latitude", value=40.7128)
    lon = st.sidebar.number_input("Longitude", value=-74.0060)

# Real-time data section
st.header("Real-Time Weather Data")
if st.button("Fetch Real-Time Data"):
    weather_data = get_real_time_weather(lat, lon)
    if weather_data:
        st.session_state.weather_data = pd.DataFrame([weather_data])
        st.success("Data fetched successfully!")
        st.write("Latest Weather Data:", st.session_state.weather_data)

# Load and prepare data
data = load_data()
features = ['hour', 'cloudCoverage', 'airTemperature', 'windSpeed']
X = data[features]
y = data['solar_potential']

# Train and compare models
if 'weather_data' in st.session_state:
    real_time_df = st.session_state.weather_data
    real_time_df['hour'] = pd.to_datetime(real_time_df['timestamp']).dt.hour
    X_real_time = real_time_df[features]
    trained_models, predictions, X_test, y_test, scaler = train_models(X, y)
    
    # Model performance
    st.header("Model Performance Comparison")
    col1, col2 = st.columns(2)
    for name, pred in predictions.items():
        with col1 if name == "Linear Regression" else col2:
            mae = mean_absolute_error(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            st.write(f"**{name}**")
            st.write(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")
        
        # Classification metrics
        y_test_binary = (y_test > y_test.median()).astype(int)
        pred_binary = (pred > y_test.median()).astype(int)
        f1 = f1_score(y_test_binary, pred_binary)
        roc_auc = roc_auc_score(y_test_binary, pred)
        cm = confusion_matrix(y_test_binary, pred_binary)
        with col1 if name == "Linear Regression" else col2:
            st.write(f"F1-score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
            st.write("Confusion Matrix:\n", cm)

    # Real-time prediction
    st.header("Real-Time Solar Potential Predictions")
    if not X_real_time.empty:
        X_real_time_scaled = scaler.transform(X_real_time)
        real_time_preds = {name: model.predict(X_real_time_scaled)[0] for name, model in trained_models.items()}
        for name, pred in real_time_preds.items():
            st.write(f"{name}: {pred:.4f} (Solar Potential)")
            co2_reduction = estimate_co2_reduction(pred)
            st.write(f"Estimated CO2 Reduction: {co2_reduction:.2f} kg/day")

  # Visualization
st.header("Prediction vs Actual (Test Set)")
if 'weather_data' not in st.session_state:
    st.warning("Please fetch real-time weather data to enable visualization.")
elif len(X_test) == 0 or len(y_test) == 0:
    st.error("Test data is empty. Cannot generate visualization.")
else:
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, pred in predictions.items():
            if len(pred) == len(y_test):
                ax.scatter(y_test, pred, label=name, alpha=0.5)
            else:
                st.warning(f"Prediction length mismatch for {name}")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        ax.set_xlabel('Actual Solar Potential')
        ax.set_ylabel('Predicted Solar Potential')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")

    # Sustainability Impact
    st.header("Sustainability Impact (SDG 7)")
    total_potential = np.mean(list(real_time_preds.values()))
    total_co2_reduction = estimate_co2_reduction(total_potential) * 365  # Annual estimate
    st.write(f"Annual CO2 Reduction Potential: {total_co2_reduction:.2f} kg/year")
    st.write("This model supports SDG 7 by optimizing solar energy use, reducing fossil fuel dependency, and promoting equitable energy access across regions.")

# Footer
st.sidebar.write("Developed for SDG 7: Affordable and Clean Energy")
st.sidebar.write("Data Source: weather.csv, Real-Time: OpenWeatherMap API")