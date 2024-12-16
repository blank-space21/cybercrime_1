from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and label encoders
model = joblib.load('crime_prediction_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    data = request.form
    crime_location = data['crime_location']
    time_of_day = data['time_of_day']
    day_of_week = data['day_of_week']
    weather_condition = data['weather_condition']
    suspect_gender = data['suspect_gender']
    weapon_used = data['weapon_used']
    crime_severity = data['crime_severity']

    # Prepare input data for prediction
    input_data = pd.DataFrame([[crime_location, time_of_day, day_of_week,  weather_condition, suspect_gender, weapon_used, crime_severity]],
                              columns=['Crime_Location', 'Time_of_Day', 'Day_of_Week', 'Weather_Condition', 'Suspect_Gender', 'Weapon_Used','Crime_Severity'])

    # Encode the input data using the saved label encoders
    for column in input_data.columns:
        le = label_encoders[column]
        input_data[column] = le.transform(input_data[column])

    # Make the prediction
    prediction = model.predict(input_data)

    predicted_crime_type = prediction[0]  # This will be 'High', 'Medium', or 'Low'

    # Return the prediction as a JSON response
    return render_template('index.html', predicted_crime_type=predicted_crime_type)

if __name__ == '__main__':
    app.run(debug=True)