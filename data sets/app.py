from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained RandomForest model
model = joblib.load('coral_health_model.pkl')

# Region encoding dictionary
region_dict = {'Upper Keys': 0, 'Middle Keys': 1, 'Lower Keys': 2}

def get_status_and_reason(pred_density):
    # Debugging: Print the predicted density
    print(f"Predicted Coral Density: {pred_density}")

    # Adjust thresholds if the model predicts lower densities
    if pred_density >= 10:
        return "Good", "Coral density is high due to good species richness and stable temperature."
    elif 5 <= pred_density < 10:
        return "Moderate", "Moderate coral health — check temperature or species diversity."
    else:
        return "Poor", "Low coral density — possibly due to high temperature or low richness."

@app.route('/')
def home():
    return render_template('index.html')  # Your input form page

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the user
    year = int(request.form['year'])
    richness = float(request.form['richness'])
    temp = float(request.form['temp'])
    region = request.form['region']
    
    # Encode the region
    region_encoded = region_dict.get(region)

    # Prepare the input data for prediction
    input_data = np.array([[year, richness, temp, region_encoded]])
    
    # Predict coral density using the trained RandomForest model
    predicted_density = model.predict(input_data)[0]
    
    # Print the predicted value for debugging
    print(f"Predicted Coral Density: {predicted_density}")

    # Get the status and reason based on the predicted coral density
    status, reason = get_status_and_reason(predicted_density)

    # Return the result in the result.html template
    return render_template('result.html',
                           predicted_density=round(predicted_density, 2),
                           status=status,
                           reason=reason,
                           year=year,
                           richness=richness,
                           temp=temp,
                           region=region)

if __name__ == '_main_':
    app.run(debug=True)