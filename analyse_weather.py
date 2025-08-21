# Import the pandas library, which is essential for data analysis in Python
import pandas as pd

# --- Step 1: Load the Dataset ---
# This line reads your CSV file into a pandas DataFrame.
# A DataFrame is like a smart spreadsheet that Python can work with.
# Make sure 'bangalore_weather.csv' is in the same folder as this script.
try:
    df = pd.read_csv('bangalore_weather.csv')
    print("✅ CSV file loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'bangalore_weather.csv' not found.")
    print("Please make sure the CSV file is in the same folder as this script.")
    exit() # Stop the script if the file isn't found

# --- Step 2: Initial Inspection ---
# We'll print some basic information to understand our data.

# 1. Print the first 5 rows to get a quick look at the columns and values.
print("\n--- First 5 Rows ---")
print(df.head())

# 2. Get a summary of the dataset (column names, data types, non-null values).
print("\n--- Data Info ---")
df.info()

# 3. Get basic statistics (mean, min, max, etc.) for numerical columns.
print("\n--- Statistical Summary ---")
print(df.describe())


# --- Step 3: Data Cleaning ---
# Here we fix any issues we found during inspection.

# 1. Convert the 'Date' column from text to a proper datetime format.
# This is crucial for working with time-series data.
print("\n🔄 Converting 'Date' column to datetime format...")
df['Date'] = pd.to_datetime(df['Date'])
print("✅ 'Date' column converted.")

# 2. Check for any missing values in each column.
print("\n--- Missing Values Before Cleaning ---")
print(df.isnull().sum())

# 3. Fill any missing values using the 'forward fill' method.
# This replaces a missing value with the last known value from the row above it.
print("\n🔄 Filling any missing values...")
df.fillna(method='ffill', inplace=True)
print("✅ Missing values filled.")

# Verify that there are no more missing values.
print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())


# --- Step 4: Feature Engineering ---
# We create new, useful columns from our existing data.

print("\n🔄 Creating new features (Year, Month, Day)...")
# Extract the Year, Month, and Day from the 'Date' column into their own columns.
# This will help our model understand seasonality.
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
print("✅ New features created.")

# Display the first 5 rows again to see our new columns.
print("\n--- Data with New Features ---")
print(df.head())

print("\n🎉 Data preparation and analysis complete! Your data is now clean and ready.")
