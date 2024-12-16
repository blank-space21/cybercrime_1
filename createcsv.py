import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Feature definitions
indian_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
times_of_day = ['Morning', 'Afternoon', 'Evening', 'Night']
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
crime_types = ['Theft', 'Assault', 'Robbery', 'Murder']
weather_conditions = ['Sunny', 'Rainy', 'Foggy', 'Cloudy']
genders = ['Male', 'Female']
weapons = ['Yes', 'No']
crime_severity = ['Low', 'Medium', 'High']

# Create synthetic data
data = pd.DataFrame({
    'Crime_Location': np.random.choice(indian_cities, n_samples),
    'Time_of_Day': np.random.choice(times_of_day, n_samples),
    'Day_of_Week': np.random.choice(days_of_week, n_samples),
    'Crime_Type': np.random.choice(crime_types, n_samples),
    'Weather_Condition': np.random.choice(weather_conditions, n_samples),
    'Suspect_Gender': np.random.choice(genders, n_samples),
    'Weapon_Used': np.random.choice(weapons, n_samples),
    'Crime_Severity': np.random.choice(crime_severity, n_samples),
})

# Relationships between features and target
# For example: 'Murder' is more likely to be 'High' severity, 'Theft' is more likely to be 'Low' severity
conditions = [
    (data['Crime_Type'] == 'Murder'),
    (data['Crime_Type'] == 'Robbery'),
    (data['Crime_Type'] == 'Assault'),
    (data['Crime_Type'] == 'Theft'),
]

# Assign corresponding severity scores based on crime type
severity_values = ['High', 'High', 'Medium', 'Low']
data['Crime_Severity'] = np.select(conditions, severity_values, default='Medium')

# Save the dataset to a CSV file
data.to_csv('india_crime_dataset.csv', index=False)

print("Indian crime dataset created successfully.")
