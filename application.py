from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

app = Flask(__name__)
data=pd.read_csv('Cleaned crop.csv')


@app.route('/')
def index():
    apmc=sorted(data['APMC'].unique())
    commodity=sorted(data['Commodity'].unique())
    month=sorted(data['Month'].unique())
    district=sorted(data['district_name'].unique())
    return render_template('home.html',apmc=apmc,commodity=commodity,month=month,district=district)


    
    #arr = np.array([[APMC, Commodity, Year, Month,Quantities,district_name]])
    #pred=model.predict(pd.DataFrame([[APMC, Commodity, Year, Month,Quantities,district_name]],columns=['APMC','Commodity','Year','Month','arrivals_in_qtl','district_name']))
    #pred = model.predict(arr)
   
@app.route('/predict',methods=['POST'])
def predict():
    apmc=request.form.get('apmc')
    commodity=request.form.get('commodity')
    year=request.form.get('year')
    month=request.form.get('month')
    quantity=request.form.get('quantity')
    district=request.form.get('district')
    pred=model.predict(pd.DataFrame([[apmc, commodity, year, month,quantity,district]],columns=['APMC','Commodity','Year','Month','arrivals_in_qtl','district_name']))
    return str(np.round(pred,2))


if __name__ == "__main__":
    app.run(debug=True)




