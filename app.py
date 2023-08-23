import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template
from Utilities.TransformFeatures import *

app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template('home.html')

@app.route("/", methods=['POST'])
def predict():

    """ Selected feature are 'tenure', 'MonthlyCharges', 'SeniorCitizen', 
        'Partner', 'Dependents', 'OnlineSecurity', 'TechSupport', 
        'PaperlessBilling', 'Contract', 'PaymentMethod' """

    monthlyCharges = float(request.form['MonthlyCharges'])
    tenure = float(request.form['Tenure'])
    seniorCitizen = request.form['SeniorCitizen']
    partner= request.form['Partner']
    dependents = request.form['Dependents']
    onlineSecurity = request.form['OnlineSecurity']
    techSupport = request.form['TechnicalSupport']
    paperlessBilling = request.form['PaperlessBilling']
    contract = request.form['Contract']
    paymentMethod = request.form['PaymentMethod']

    model_filename = 'model.pkl'
    load_model = pickle.load(open(model_filename, 'rb'))
    dd = {'tenure':tenure, 'MonthlyCharges':monthlyCharges, 'SeniorCitizen': seniorCitizen, 
          'Partner': partner, 'Dependents': dependents, 'OnlineSecurity': onlineSecurity, 'TechSupport': techSupport, 
          'PaperlessBilling': paperlessBilling, 'Contract': contract, 'PaymentMethod': paymentMethod}

    data=transform_features(dd)

    df1=pd.DataFrame(data, index=[0])
    single = load_model.predict(df1)
    probability = load_model.predict_proba(df1)[:,1]
    # probability = probability*100

    if single == 1:
        op1 = "This customer is likely to be churned."
        op2 = f"It's probability is {np.round(probability[0], 2)}"
    else:
        op1 = "This customer is likely to continue!"
        op2 = f"It's probability is {np.round(1 - probability[0], 2)}"

    return render_template("home.html", op1=op1, op2=op2,
                            monthlyCharges = monthlyCharges, tenure = tenure,
                            seniorCitizen = seniorCitizen, partner = partner,
                            dependents = dependents, onlineSecurity = onlineSecurity,
                            techSupport = techSupport, paperlessBilling = paperlessBilling,
                            contract = contract, paymentMethod = paymentMethod)


if __name__ == '__main__':
    app.run()