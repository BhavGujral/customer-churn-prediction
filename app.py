from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from form
    input_data = [float(x) for x in request.form.values()]
    final_input = np.array(input_data).reshape(1, -1)
    
    # Scale and predict
    scaled_input = scaler.transform(final_input)
    prediction = model.predict(scaled_input)
    
    result = "Churn" if prediction[0] == 1 else "Not Churn"
    return render_template('result.html', prediction_text=f"The customer is likely to {result}.")

if __name__ == '__main__':
    app.run(debug=True)