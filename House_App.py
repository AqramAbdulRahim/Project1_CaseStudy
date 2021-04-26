from flask import Flask, render_template,  redirect, url_for, session
from flask_wtf import FlaskForm 
from wtforms import TextField,SubmitField 
from wtforms.validators import NumberRange 
from joblib import dump, load


import numpy as np  



def return_prediction(model, scaler, sample_json):
    
    Lot_Area = sample_json['Lot Area']
    Overal_Qual = sample_json['Overal Qual']
    Year_Built = sample_json['Year Built']
    Year_Remod = sample_json['Year Remod']
    Mas_Vnr_Area = sample_json['Mas Vnr Area']
    BsmtFin_SF1 = sample_json['BsmtFin SF 1']
    Total_Bsmt_SF = sample_json['Total Bsmt SF']
    First_Flr_SF = sample_json['First Flr SF']
    Gr_Liv_Area = sample_json['Gr Liv Area']
    Fireplaces = sample_json['Fireplaces']
    Garage_Area = sample_json['Garage Area']
    Neighborhood_NridgHt = sample_json['Neighborhood_NridHt']
    Bsmt_Exposure_Gd = sample_json['BsmtExposure_Gd']
    Sale_Type_New = sample_json['Sale Type_New']
    
    house_price = [[Lot_Area, Overal_Qual, Year_Built, Year_Remod, Mas_Vnr_Area, BsmtFin_SF1,
                   Total_Bsmt_SF, First_Flr_SF, Gr_Liv_Area, Fireplaces, Garage_Area,
                   Neighborhood_NridgHt, Bsmt_Exposure_Gd, Sale_Type_New]]
    
    house_price = scaler.transform(house_price)
    
    prediction = model.predict(house_price)
    
    return prediction[0]


app = Flask(__name__)

app.config['SECRET_KEY'] = 'mysecretkey' 


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
house_model = load("AMES_Housing_model.h5")
house_scaler = load("AMES_Housing_scaler.pkl")


# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class HouseForm(FlaskForm): 
    Lot_Area = TextField('Lot Area') 
    Overal_Qual = TextField('Overal Qual') 
    Year_Built = TextField('Year Built')
    Year_Remod = TextField('Year Remod')
    Mas_Vnr_Area = TextField('Mas Vnr Area') 
    BsmtFin_SF_1 = TextField('BsmtFin SF 1') 
    Total_Bsmt_SF = TextField('Total Bsmt SF')
    First_Flr_SF = TextField('First Flr SF')
    Gr_Liv_Area = TextField('Gr Liv Area') 
    Fireplaces = TextField('Fireplaces') 
    Garage_Area = TextField('Garage Area')
    Neighborhood_NridHt = TextField('Neighborhood NridHt')
    BsmtExposure_Gd = TextField('BsmtExposure Gd') 
    Sale_Type_New = TextField('Sale Type New') 
   
    submit = SubmitField('Analyze') 



@app.route('/', methods=['GET', 'POST'])
def index():

    form = HouseForm()
    if form.validate_on_submit(): 
        # Grab the data from the breed on the form.

        session['Lot_Area'] = form.Lot_Area.data
        session['Overal_Qual'] = form.Overal_Qual.data
        session['Year_Built'] = form.Year_Built.data
        session['Year_Remod'] = form.Year_Remod.data
        session['Mas_Vnr_Area'] = form.Mas_Vnr_Area.data
        session['BsmtFin_SF_1'] = form.BsmtFin_SF_1.data
        session['Total_Bsmt_SF'] = form.Total_Bsmt_SF.data
        session['First_Flr_SF'] = form.First_Flr_SF.data
        session['Gr_Liv_Area'] = form.Gr_Liv_Area.data
        session['Fireplaces'] = form.Fireplaces.data
        session['Garage_Area'] = form.Garage_Area.data
        session['Neighborhood_NridHt'] = form.Neighborhood_NridHt.data
        session['BsmtExposure_Gd'] = form.BsmtExposure_Gd.data
        session['Sale_Type_New'] = form.Sale_Type_New.data

        return redirect(url_for("prediction"))


    return render_template('househome.html', form=form) 


@app.route('/houseprediction')
def prediction():

    content = {}

    content['Lot Area'] = float(session['Lot_Area']) 
    content['Overal Qual'] = float(session['Overal_Qual'])
    content['Year Built'] = float(session['Year_Built'])
    content['Year Remod'] = float(session['Year_Remod'])
    content['Mas Vnr Area'] = float(session['Mas_Vnr_Area'])
    content['BsmtFin SF 1'] = float(session['BsmtFin_SF_1'])
    content['Total Bsmt SF'] = float(session['Total_Bsmt_SF'])
    content['First Flr SF'] = float(session['First_Flr_SF'])
    content['Gr Liv Area'] = float(session['Gr_Liv_Area'])
    content['Fireplaces'] = float(session['Fireplaces'])
    content['Garage Area'] = float(session['Garage_Area'])
    content['Neighborhood_NridHt'] = float(session['Neighborhood_NridHt'])
    content['BsmtExposure_Gd'] = float(session['BsmtExposure_Gd'])
    content['Sale Type_New'] = float(session['Sale_Type_New'])

    results = return_prediction(model=house_model,scaler=house_scaler,sample_json=content) 

    return render_template('houseprediction.html',results=results)


if __name__ == '__main__':
    app.run(debug=True)