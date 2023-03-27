from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np

import pickle

from sklearn.ensemble import RandomForestClassifier


######
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/credit_risk_prediction',methods=['POST'])
def credit_risk_prediction():
    if request.method == 'POST':
        age = int(request.form['age'])
        income = int(request.form['income'])
        debt = int(request.form['debt'])
        credit_score = int(request.form['credit_score'])
        is_employed = int(request.form['is_employed'])
        data = [[age, income, debt, credit_score, is_employed]]
        prediction = forest.predict(data)
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
	
	##Initialize forest	
	forest = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_split=10, min_samples_leaf=5)

with open('model/Model.pkl', 'rb') as f:
	    forest = pickle.load(f)

app.run(host='0.0.0.0',port=5000, debug=True)




