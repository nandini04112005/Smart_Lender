from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')


@app.route('/')
def welcome():
    return render_template('welcome.html')


@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:

        features = [
            float(request.form['Gender']),
            float(request.form['Married']),
            float(request.form['Dependents']),
            float(request.form['Education']),
            float(request.form['Self_Employed']),
            float(request.form['ApplicantIncome']),
            float(request.form['CoapplicantIncome']),
            float(request.form['LoanAmount']),
            float(request.form['Loan_Amount_Term']),
            float(request.form['Credit_History']),
            float(request.form['Property_Area'])
        ]

        features = np.array(features).reshape(1, -1)

        features = scaler.transform(features)

        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "✅ Loan Approved"
        else:
            result = "❌ Loan Rejected"

        return render_template(
            'result.html',
            prediction=result
        )

    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(debug=True)