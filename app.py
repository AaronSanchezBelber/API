from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np

import pickle

from sklearn.ensemble import RandomForestClassifier

model = None
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

######
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/credit_risk_prediction',methods=['POST'])
def credit_risk_prediction():
    if request.method == 'POST':
        person_age = int(request.form['age'])
        person_income = int(request.form['income'])
        person_emp_length = int(request.form['debt'])
        loan_amnt = int(request.form['credit_score'])
        cb_person_cred_hist_length = int(request.form['cb_person_cred_hist_length'])
        # loan_percent_income = loan_amnt/person_income
        # loan_int_rate = 8
        cb_person_default_on_file = 0
        person_home_ownership_MORTGAGE = 0
        person_home_ownership_OWN = 0
        person_home_ownership_RENT = 1
        loan_intent_DEBTCONSOLIDATION = 0
        loan_intent_EDUCATION = 0
        loan_intent_HOMEIMPROVEMENT = 0
        loan_intent_MEDICAL = 0
        loan_intent_PERSONAL = 1
        loan_intent_VENTURE = 0
        loan_grade_A = 0
        loan_grade_B = 0
        loan_grade_C = 1
        loan_grade_D = 0
        loan_grade_E = 0


        data = [[person_age, person_income, person_emp_length, loan_amnt, cb_person_default_on_file, cb_person_cred_hist_length, person_home_ownership_MORTGAGE,
                  person_home_ownership_OWN, person_home_ownership_RENT, loan_intent_DEBTCONSOLIDATION, loan_intent_EDUCATION,
                  loan_intent_HOMEIMPROVEMENT, loan_intent_MEDICAL, loan_intent_PERSONAL, loan_intent_VENTURE, loan_grade_A,
                  loan_grade_B, loan_grade_C, loan_grade_D, loan_grade_E]]
        
        prediction = forest.predict(data)
    return render_template('result.html', prediction= prediction)


if __name__ == '__main__':
	
	##Initialize forest	
	forest = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_split=10, min_samples_leaf=5)

with open('model/Model.pkl', 'rb') as f:
	    forest = pickle.load(f)

app.run(host='0.0.0.0',port=5000, debug=True)




