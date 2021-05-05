from flask import Flask, render_template,  redirect, url_for, session
from flask_wtf import FlaskForm 
from wtforms import TextField,SubmitField 
from wtforms.validators import NumberRange 
from joblib import dump, load


import numpy as np  



def return_prediction(model, scaler, sample_json):
    
    Lot_Frontage = sample_json['Lot Frontage']
    Lot_Area = sample_json['Lot Area']
    Overall_Qual = sample_json['Overall Qual']
    Year_Built = sample_json['Year Built']
    Year_Remod = sample_json['Year Remod/Add']
    Exter_Qual = sample_json['Exter Qual']
    Bsmt_Qual = sample_json['Bsmt Qual']
    Bsmt_Exposure = sample_json['Bsmt Exposure']
    BsmtFin_SF_1 = sample_json['BsmtFin SF 1']
    Total_Bsmt_SF = sample_json['Total Bsmt SF']
    First_Flr_SF = sample_json['1st Flr SF']
    Gr_Liv_Area = sample_json['Gr Liv Area']
    Kitchen_Qual = sample_json['Kitchen Qual']
    Fireplaces = sample_json['Fireplace Qu']
    Garage_Finish = sample_json['Garage Finish']
    Garage_Cars = sample_json['Garage Cars']
    Garage_Area = sample_json['Garage Area']
    Neighborhood_NridgHt = sample_json['Neighborhood_NridgHt']
    Sale_Condition_Partial = sample_json['Sale Condition_Partial']
    
    house_price = [[Lot_Frontage, Lot_Area, Overall_Qual, Year_Built, Year_Remod, Exter_Qual, Bsmt_Qual, Bsmt_Exposure, BsmtFin_SF_1,
                   Total_Bsmt_SF, First_Flr_SF, Gr_Liv_Area, Kitchen_Qual, Fireplaces, Garage_Finish, Garage_Cars, Garage_Area,
                   Neighborhood_NridgHt, Sale_Condition_Partial]]
    
    house_price = scaler.transform(house_price)
    
    prediction = model.predict(house_price)
    
    return prediction[0]


app = Flask(__name__)

app.config['SECRET_KEY'] = 'mysecretkey' 


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
house_model = load("HouseSales2_model.h5")
house_scaler = load("HouseSales2_scaler.pkl")


# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class HouseForm(FlaskForm): 
    Lot_Frontage = TextField('Lot Frontage')
    Lot_Area = TextField('Lot Area')
    Overall_Qual = TextField('Overall Qual')
    Year_Built = TextField('Year Built')
    Year_Remod = TextField('Year Remod/Add')
    Exter_Qual = TextField('Exter Qual')
    Bsmt_Qual = TextField('Bsmt Qual')
    Bsmt_Exposure = TextField('Bsmt Exposure')
    BsmtFin_SF_1 = TextField('BsmtFin SF 1')
    Total_Bsmt_SF = TextField('Total Bsmt SF')
    First_Flr_SF = TextField('1st Flr SF')
    Gr_Liv_Area = TextField('Gr Liv Area')
    Kitchen_Qual = TextField('Kitchen Qual')
    Fireplaces = TextField('Fireplace Qu')
    Garage_Finish = TextField('Garage Finish')
    Garage_Cars = TextField('Garage Cars')
    Garage_Area = TextField('Garage Area')
    Neighborhood_NridgHt = TextField('Neighborhood_NridgHt')
    Sale_Condition_Partial = TextField('Sale Condition_Partial')
   
    submit = SubmitField('Analyze') 
    print(Overall_Qual)

@app.route('/', methods=['GET', 'POST'])
def index():

    form = HouseForm()
    if form.validate_on_submit(): 
        # Grab the data from the breed on the form.

        session['Lot_Frontage'] = form.Lot_Frontage.data
        session['Lot_Area'] = form.Lot_Area.data
        session['Overall_Qual'] = form.Overall_Qual.data
        session['Year_Built'] = form.Year_Built.data
        session['Year_Remod'] = form.Year_Remod.data
        session['Exter_Qual'] = form.Exter_Qual.data
        session['Bsmt_Qual'] = form.BsmtFin_SF_1.data
        session['Bsmt_Exposure'] = form.Bsmt_Exposure.data
        session['BsmtFin_SF_1'] = form.BsmtFin_SF_1.data
        session['Total_Bsmt_SF'] = form.Total_Bsmt_SF.data
        session['First_Flr_SF'] = form.First_Flr_SF.data
        session['Gr_Liv_Area'] = form.Gr_Liv_Area.data
        session['Kitchen_Qual'] = form.Kitchen_Qual.data
        session['Fireplaces'] = form.Fireplaces.data
        session['Garage_Finish'] = form.Garage_Finish.data
        session['Garage_Cars'] = form.Garage_Cars.data
        session['Garage_Area'] = form.Garage_Area.data
        session['Neighborhood_NridgHt'] = form.Neighborhood_NridgHt.data
        session['Sale_Condition_Partial'] = form.Sale_Condition_Partial.data

        return redirect(url_for("prediction"))


    return render_template('house_input.html', form=form) 


@app.route('/predict')
def prediction():

    content = {}

    content['Lot Frontage'] = float(session['Lot_Frontage']) 
    content['Lot Area'] = float(session['Lot_Area']) 
    content['Overall Qual'] = float(session['Overall_Qual'])
    content['Year Built'] = float(session['Year_Built'])
    content['Year Remod/Add'] = float(session['Year_Remod'])
    content['Exter Qual'] = float(session['Exter_Qual'])
    content['Bsmt Qual'] = float(session['Bsmt_Qual'])
    content['Bsmt Exposure'] = float(session['Bsmt_Exposure'])
    content['BsmtFin SF 1'] = float(session['BsmtFin_SF_1'])
    content['Total Bsmt SF'] = float(session['Total_Bsmt_SF'])
    content['1st Flr SF'] = float(session['First_Flr_SF'])
    content['Gr Liv Area'] = float(session['Gr_Liv_Area'])
    content['Kitchen Qual'] = float(session['Kitchen_Qual'])
    content['Fireplace Qu'] = float(session['Fireplaces'])
    content['Garage Finish'] = float(session['Garage_Finish'])
    content['Garage Cars'] = float(session['Garage_Cars'])
    content['Garage Area'] = float(session['Garage_Area'])
    content['Neighborhood_NridgHt'] = float(session['Neighborhood_NridgHt'])
    content['Sale Condition_Partial'] = float(session['Sale_Condition_Partial'])

    results = return_prediction(model=house_model,scaler=house_scaler,sample_json=content) 

    return render_template('predictions.html',results=round(results,2))


if __name__ == '__main__':
    app.run(debug=True)