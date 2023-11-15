from pickle import TRUE
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open('LinearRegressionModel.pkl','rb'))
car = pd.read_csv('Cleaned_Car_data.csv')


@app.route('/', methods=['GET','POST'])
def hello_world():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')

    return render_template('home.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)


@app.route("/predict", methods=['GET'])
def predict_data():    
    company = request.args.get('company')
    cmodel = request.args.get('cmodel')
    year = request.args.get('year')
    kml = request.args.get('kml')
    fueltype  = request.args.get('fueltype')

    prediction = model.predict(pd.DataFrame(
                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], 
                data=np.array([cmodel,company,year,kml,fueltype]).reshape(1, 5)))
    
    return render_template('predict.html', result=str(np.round(prediction[0],2)))
    

@app.route("/about")
def about():
    return render_template('about.html')


if __name__=="__main__":
    app.run(debug=True,port=8000)
    
    
    