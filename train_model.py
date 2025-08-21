# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import joblib # Used for saving the trained model

# --- Step 1: Load and Prepare the Data ---
try:
    df = pd.read_csv('bangalore_weather.csv')
    print("✅ CSV file loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'bangalore_weather.csv' not found.")
    exit()

# Perform the same cleaning and feature engineering steps
df['Date'] = pd.to_datetime(df['Date'])
df.fillna(method='ffill', inplace=True)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
print("✅ Data prepared for modeling.")

# --- Step 2: Define Features (X) and Target (y) ---
# We want to predict the max temperature based on the date.
# The features are the inputs our model will learn from.
features = ['Year', 'Month', 'Day']
X = df[features]

# The target is the value we want to predict.
y = df['Temp Max']

# --- Step 3: Split Data into Training and Testing Sets ---
# We'll train the model on 80% of the data and test its performance on the remaining 20%.
# We set shuffle=False because this is time-series data; we want to predict the future.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"✅ Data split into training and testing sets.")
print(f"   - Training set size: {len(X_train)} samples")
print(f"   - Testing set size: {len(X_test)} samples")

# --- Step 4: Train the Machine Learning Model ---
# We are using a 'Random Forest Regressor', which is a powerful and popular model.
# n_estimators=100 means it uses 100 "decision trees" to make a more robust prediction.
# random_state=42 ensures that we get the same result every time we run the script.
print("\n🔄 Training the Random Forest model... (This might take a moment)")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all available CPU cores
model.fit(X_train, y_train)
print("✅ Model training complete.")

# --- Step 5: Evaluate the Model's Performance ---
# Now we use our trained model to make predictions on the test data (dates it has never seen).
print("\n🔄 Making predictions on the test set...")
predictions = model.predict(X_test)
print("✅ Predictions made.")

# Calculate the Mean Absolute Error (MAE).
# This tells us, on average, how many degrees Celsius our predictions are off by.
mae = mean_absolute_error(y_test, predictions)
print(f"\n--- Model Performance ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} °C")
print(f"-> This means, on average, our model's temperature predictions are off by about {mae:.2f} degrees Celsius.")

# --- Step 6: Compare Predictions with Actual Values ---
# Let's look at a few examples to see how well it did.
comparison_df = pd.DataFrame({'Actual Temp Max': y_test, 'Predicted Temp Max': predictions})
print("\n--- Sample of Predictions vs. Actual Values ---")
print(comparison_df.head(10)) # Show the first 10 predictions

# --- Step 7: Save the Trained Model ---
# We save the trained model to a file so we can use it later in our website
# without having to retrain it every time.
model_filename = 'weather_model.joblib'
joblib.dump(model, model_filename)
print(f"\n✅ Model saved to '{model_filename}'.")

print("\n🎉 Model training and evaluation process complete!")
