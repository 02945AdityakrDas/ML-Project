from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('LogisticRegression_model_HeartDisease.pkl')
scaler = StandardScaler()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        age = float(request.form['age'])
        sex_male = float(request.form['sex_male'])
        cigs_per_day = float(request.form['cigs_per_day'])
        tot_chol = float(request.form['tot_chol'])
        sys_bp = float(request.form['sys_bp'])
        glucose = float(request.form['glucose'])

        # Combine features into a single array
        features = np.array(
            [[age, sex_male, cigs_per_day, tot_chol, sys_bp, glucose]])
        features = scaler.fit_transform(features)

        # Predict using the model
        prediction = model.predict(features)[0]

        # Interpret the result
        result_message = "Patient is at risk of CHD" if prediction == 1 else "Patient is not at risk of CHD"
        return render_template('index.html', prediction=result_message)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
