import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model and scaler
with open('model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def preprocess_new_data_array(new_data_array, scaler):
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    new_data_df = pd.DataFrame(new_data_array, columns=column_names)
    new_data_df['DiabetesPedigreeFunction'] = np.log1p(new_data_df['DiabetesPedigreeFunction'])
    X_new = new_data_df.select_dtypes(include=["int64", "float64"])
    X_new_scaled = scaler.transform(X_new)
    return X_new_scaled

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('GDM.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction_text = None
    form_data = request.form.to_dict()
    
    try:
        features = [float(form_data.get(feature, 0)) for feature in [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]]
        features = np.array(features).reshape(1, -1)
        X_new_scaled = preprocess_new_data_array(features, scaler)
        prediction = xgb_model.predict(X_new_scaled)[0]
        prediction_text = 'Diabetes' if prediction == 1 else 'No Diabetes'
    except Exception as e:
        prediction_text = f"Error: {e}"
    
    return render_template('index.html', prediction_text=prediction_text, **form_data)

if __name__ == "__main__":
    app.run(debug=True)
