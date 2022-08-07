from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("LinearRegressionModel.pkl","rb"))

cars = pd.read_csv('Cleaned_cars_data.csv')

@app.route('/')
def index():
    companies = sorted(cars['company'].unique())
    car_model = sorted(cars['name'].unique())
    year = sorted(cars['year'].unique(),reverse=True)
    fuel_type = cars['fuel_type'].unique()
    companies.insert(0,"Select Company")            # Inserting "Select Company" in 0th position of the companies category
    return render_template('index.html',companies=companies,car_models=car_model,years=year,fuel_types=fuel_type)

@app.route('/predict',methods=['POST'])
def predict():
    company = request.form.get('company_name')
    car_model = request.form.get('car_model_name')
    year = request.form.get('year_name')
    fuel_type = request.form.get('fuel_type_name')
    kms_driven = request.form.get('kilo_driven_name')
    prediction = model.predict(pd.DataFrame([[car_model,company,year,kms_driven,fuel_type]],columns=['name','company','year','kms_driven','fuel_type']))

    return str(np.round(prediction[0],2))

if __name__=="__main__":
    app.run(debug=True)
