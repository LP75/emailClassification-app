from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd 
import pickle

app = Flask(__name__)

cv = pickle.load(open("models/cv.pkl", 'rb'))
clf = pickle.load(open("models/clf.pkl", 'rb'))

@app.route('/')
def home():
    return render_template("input.html")

@app.route('/predict',methods=['POST'])
def predict():
    email = request.form['emailText']
    tokenized_email = cv.transform([email])
    prediction = clf.predict(tokenized_email)
    return render_template('input.html', resultat=prediction[0])

@app.route('/api/predict',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    email = data['email']
    tokenized_email = cv.transform([email])
    prediction = clf.predict(tokenized_email)
    result = {
        "prediction": int(prediction[0]),
        "label": "spam" if prediction[0] == 1 else "non spam"
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run()