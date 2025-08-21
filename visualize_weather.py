# Import necessary libraries
# pandas for data handling, matplotlib and seaborn for plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load and Prepare the Data ---
# We'll repeat the cleaning steps to ensure we're working with the clean dataset.
try:
    df = pd.read_csv('bangalore_weather.csv')
    print("✅ CSV file loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'bangalore_weather.csv' not found.")
    exit()

# Convert 'Date' column to datetime and fill any missing values
df['Date'] = pd.to_datetime(df['Date'])
df.fillna(method='ffill', inplace=True)

# Create Year and Month columns for grouping
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
print("✅ Data prepared for visualization.")

# --- Step 2: Create Visualizations ---

# --- Visualization 1: Average Max Temperature Over the Years ---
# This plot helps us see long-term trends, like global warming.
print("\n🔄 Generating plot 1: Average Maximum Temperature Trend...")
plt.figure(figsize=(15, 7)) # Set the size of the plot
# Group the data by year and calculate the mean of 'Temp Max' for each year
df.groupby('Year')['Temp Max'].mean().plot(kind='line', marker='o', linestyle='-')
plt.title('Average Maximum Temperature in Bangalore (1951-2024)', fontsize=16)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.grid(True)
plt.show() # Display the plot
print("✅ Plot 1 displayed.")


# --- Visualization 2: Monthly Temperature Distribution (Box Plot) ---
# This plot shows the seasonality of the weather. We can clearly see
# which months are hottest, coolest, and which have the most variable weather.
print("\n🔄 Generating plot 2: Monthly Temperature Distribution...")
plt.figure(figsize=(12, 6))
sns.boxplot(x='Month', y='Temp Max', data=df)
plt.title('Monthly Maximum Temperature Distribution in Bangalore', fontsize=16)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.xlabel('Month', fontsize=12)
# Set more descriptive labels for the x-axis
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
print("✅ Plot 2 displayed.")


# --- Visualization 3: Average Monthly Rainfall ---
# This helps us understand Bangalore's monsoon seasons.
print("\n🔄 Generating plot 3: Average Monthly Rainfall...")
plt.figure(figsize=(12, 6))
# Group by month and calculate the average rainfall
df.groupby('Month')['Rain'].mean().plot(kind='bar')
plt.title('Average Monthly Rainfall in Bangalore', fontsize=16)
plt.ylabel('Average Rainfall (mm)', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
print("✅ Plot 3 displayed.")

print("\n🎉 Visualization complete!")