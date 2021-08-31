
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
import requests
import json
import yaml
import tensorflow as tf
import pickle
from json2html import *
from functions import *
import time
from datetime import datetime

app = Flask(__name__)


# loading the configuration files
try:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print('Error reading configuration file.')


#app.config['SQLALCHEMY_DATABASE_URI']=f"postgresql://{settings.DATABASE['DB_USER']}:{settings.DATABASE['DB_PASSWORD']}@{settings.DATABASE['DB_HOST']}/{settings.DATABASE['DB_NAME']}"
app.config['SQLALCHEMY_DATABASE_URI']=f'postgresql://{config["USER"]}:{config["PASSWORD"]}@{config["HOST"]}/{config["DATABASE"]}'


db = SQLAlchemy(app)

model = tf.keras.models.load_model('best_model.epoch08-loss0.0114.hdf5')
scaler = pickle.load(open('scaler.pkl','rb'))

class History(db.Model):
    __tablename__ = config['TABLE_HISTORY']
    datetime = db.Column(db.DateTime, primary_key=True)
    PM25 = db.Column(db.Float)
    PM10 = db.Column(db.Float)
    SO2 = db.Column(db.Float)
    NO2 = db.Column(db.Float)
    CO = db.Column(db.Float)
    O3 = db.Column(db.Float)
    TEMP = db.Column(db.Float)
    PRES = db.Column(db.Float)
    DEWP = db.Column(db.Float)
    RAIN = db.Column(db.Float)
    sin_time = db.Column(db.Float)
    cos_time = db.Column(db.Float)
    wx = db.Column(db.Float)
    wy = db.Column(db.Float)
    
    def __init__(self,datetime,pm25,pm10,so2,no2,co,o3,temp,pres,dewp,rain,sin_time,cos_time,wx,wy):
        self.datetime = datetime
        self.PM25 = pm25
        self.PM10 = pm10
        self.SO2 = so2
        self.NO2 = no2
        self.CO = co
        self.O3 = o3
        self.TEMP = temp
        self.PRES = pres
        self.DEWP = dewp
        self.RAIN = rain
        self.sin_time = sin_time
        self.cos_time = cos_time
        self.wx = wx
        self.wy = wy

class Prediction(db.Model):
    __tablename__ = config['TABLE_PREDICTIONS']
    datetime = db.Column(db.DateTime, primary_key=True)
    PM25 = db.Column(db.Float)
    
    def __init__(self, datetime, pm25):
        self.datetime = datetime
        self.PM25 = pm25
        
@app.route('/', methods=['GET','POST'])
def root():
    return render_template('index.html', href='static/thumbnail.jpg')        

    
@app.route('/current_conditions', methods=['GET','POST'])
def get_current_conditions():
    current_inputs = get_inputs()
    
    new_input = History(current_inputs[0]['datetime'],current_inputs[0]['PM25'],current_inputs[0]['PM10'],
                        current_inputs[0]['SO2'],current_inputs[0]['NO2'],current_inputs[0]['CO'],
                        current_inputs[0]['O3'],current_inputs[0]['TEMP'],current_inputs[0]['PRES'],
                        current_inputs[0]['DEWP'],current_inputs[0]['RAIN'],current_inputs[0]['sin time'],
                        current_inputs[0]['cos time'],current_inputs[0]['wx'],current_inputs[0]['wy'])

    if not db.session.query(db.exists().where(History.datetime == current_inputs[0]['datetime'])).scalar():
        db.session.add(new_input)
        db.session.commit()
    return json2html.convert(json=current_inputs[1])

@app.route('/history', methods=['GET','POST'])
def get_history():
    history = reversed(History.query.order_by(History.datetime.desc()).limit(72).all())
    response = [{
        'datetime': record.datetime,
        'PM2.5' : record.PM25,
        'PM10' : record.PM10,
        'SO2' : record.SO2,
        'NO2' : record.NO2,
        'CO' : record.CO,
        'O3' : record.O3,
        'TEMP' : record.TEMP,
        'PRES' : record.PRES,
        
        'DEWP' : record.DEWP,
        'RAIN' : record.RAIN,
        'sin time' : record.sin_time,
        'cos time' : record.cos_time,
        'wx' : record.wx,
        'wy' : record.wy
        } for record in history]
    
    j = {'Hourly history of previous three days' : response}
    return json2html.convert(json=j)
    
@app.route('/predict', methods=['GET','POST'])
def predict():
    history = reversed(History.query.order_by(History.datetime.desc()).limit(72).all())
    response = [{
        'datetime': record.datetime,
        'PM2.5' : record.PM25,
        'PM10' : record.PM10,
        'SO2' : record.SO2,
        'NO2' : record.NO2,
        'CO' : record.CO,
        'O3' : record.O3,
        'TEMP' : record.TEMP,
        'PRES' : record.PRES,
        'DEWP' : record.DEWP,
        'RAIN' : record.RAIN,
        'sin time' : record.sin_time,
        'cos time' : record.cos_time,
        'wx' : record.wx,
        'wy' : record.wy
        } for record in history]
    
    response = pd.DataFrame(response)
    response.set_index('datetime',inplace=True)
    response = np.array(response).reshape(1,72,14)
    
    prediction = predict_future(model, scaler, response)
    
    msg = print_message(prediction[1])
    
    p = Prediction(prediction[0], prediction[1])
    if not db.session.query(db.exists().where(Prediction.datetime == prediction[0])).scalar():
        db.session.add(p)
        db.session.commit()
    
    return render_template('/prediction.html', prediction=prediction[1], msg=msg)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)