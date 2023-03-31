from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import pickle
import pandas as pd
import numpy


app = Flask(__name__)
run_with_ngrok(app)  

# load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    Number_of_Pregrancies = float(request.form['Number of Pregrancies'])
    Glucose = float(request.form['Glucose'])
    Blood_Pressure = float(request.form['Blood Pressure'])
    Skin_Thickness = float(request.form['Skin Thickness'])
    Insulin = float(request.form['Insulin'])
    BMI = float(request.form['BMI'])
    Diabetes_Pedigree_Function = float(request.form['Diabetes Pedigree Function'])
    Age = float(request.form['Age'])
    
    data = [[Number_of_Pregrancies, Glucose, Blood_Pressure, Skin_Thickness, Insulin, BMI, Diabetes_Pedigree_Function, Age]]
  
    # Create the pandas DataFrame with column name is provided explicitly
    df = pd.DataFrame(data, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    result = model.predict(df)
    
    if(result == numpy.array([0])):
        result = "You don't have diabetes"
    else:
        result = "You have diabetes"
    
    return render_template('index.html', **locals())

if __name__ == '__main__':
    app.run()