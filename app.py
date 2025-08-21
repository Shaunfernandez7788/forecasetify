# Import necessary libraries
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from datetime import date, timedelta
from flask_cors import CORS
import random
import numpy as np # New import for smooth hourly data

# --- 1. Initialize the Flask App ---
app = Flask(__name__, static_folder='.', static_url_path='', template_folder='.')
CORS(app)

# --- 2. Load the Trained Model ---
try:
    model = joblib.load('weather_model.joblib')
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'weather_model.joblib' not found. Please run train_model.py first.")
    model = None

# --- 3. Create a Route for the Homepage ---
@app.route('/')
def home():
    return render_template('index.html')

# --- 4. Route for the Current Weather Theme ---
@app.route('/current-weather')
def current_weather():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    today = date.today()
    today_df = pd.DataFrame({'Year': [today.year],'Month': [today.month],'Day': [today.day]})
    predicted_temp = model.predict(today_df)[0]
    
    theme = "sunny"
    if predicted_temp < 27:
        if random.random() < 0.5: theme = "rainy"
        else: theme = "cloudy"
    elif predicted_temp >= 27:
        theme = "sunny"
    return jsonify({'theme': theme})

# --- 5. UPDATED Prediction API Endpoint with HOURLY data ---
@app.route('/predict', methods=['GET'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    today = date.today()
    future_dates = [today + timedelta(days=i) for i in range(7)]
    
    future_df = pd.DataFrame({
        'Year': [d.year for d in future_dates],
        'Month': [d.month for d in future_dates],
        'Day': [d.day for d in future_dates]
    })

    predictions_max = model.predict(future_df)
    forecast_data = []

    for i in range(len(future_dates)):
        pred_max = predictions_max[i]
        pred_min = pred_max - (random.uniform(7, 10))
        humidity = random.randint(65, 90)

        # Generate smooth hourly temperature data using a sine wave
        hours = np.arange(24)
        amplitude = (pred_max - pred_min) / 2
        mid_point = (pred_max + pred_min) / 2
        # Peaks around 3 PM (hour 15)
        hourly_temps = mid_point + amplitude * np.sin((hours - 9) * np.pi / 12)
        # Add some noise
        hourly_temps += np.random.normal(0, 0.5, 24)

        # Generate hourly humidity data
        hourly_humidity = np.linspace(humidity + 5, humidity - 10, 24)
        hourly_humidity += np.random.normal(0, 2, 24)
        
        forecast_data.append({
            'date': future_dates[i].strftime('%Y-%m-%d'),
            'day_name': future_dates[i].strftime('%A'),
            'temp_max': round(pred_max, 1),
            'temp_min': round(pred_min, 1),
            'humidity': humidity,
            'hourly_temps': [round(t, 1) for t in hourly_temps],
            'hourly_humidity': [int(h) for h in hourly_humidity]
        })

    return jsonify(forecast_data)

# --- 6. Run the Flask App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
