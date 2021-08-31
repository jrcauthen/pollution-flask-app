# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 14:24:07 2021

@author: justi
"""

import numpy as np
import pickle
import yaml
import json
import time
from datetime import datetime
import pandas as pd
import requests
#import psycopg2
import tensorflow as tf
from tensorflow import keras
from sqlalchemy import create_engine
#from psycopg2.extensions import register_adapter, AsIs



#register_adapter(np.float64, AsIs)
#register_adapter(np.float32, AsIs)


# loading the configuration files
try:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except Exception:
    print('Error reading configuration file.')
    
# connecting to the POSTGRESQL server
# cnx = psycopg2.connect(
#     database = config['DATABASE'],
#     host = config['HOST'],
#     user = config['USER'],
#     password = config['PASSWORD'])

def get_wind_vectors(degree: int, wind_speed: np.float32) -> str:
    
    '''
    Meteorological wind direction is defined as the direction from which it originates.
    For example, a northerly wind blows from the north to the south.
    Wind direction is measured in degrees clockwise from due north.
    Hence, a wind coming from the south has a wind direction of 180 degrees;
    one from the east is 90 degrees. 
    '''
    
    wx = wind_speed * np.cos(degree/360)
    wy = wind_speed * np.sin(degree/360)
    
    return wx, wy

def K_to_C(k: np.float32) -> np.float64:
    
    '''
    Convert Kelvin to Celsius
    '''

    return np.float64(k - 273.15)
        

def get_inputs() -> pd.Series:
    
    current_time = int(time.time())
    previous_three_days = current_time - 86400*3 # number of seconds in one day = 86400
    
    API_key = config['API_KEY']
    latitude = '39.9497'
    longitude = '116.3891'
    API_url_history = f'https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={latitude}&lon={longitude}&dt={previous_three_days}&appid={API_key}'
    API_url_current = f'https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={API_key}'
    
    # get weather data
    response_current = requests.get(API_url_current)
    response_history = requests.get(API_url_history)
    weather_data_current = json.loads(response_current.text)
    weather_data_history = json.loads(response_history.text)

    # scrape page for pollutant info
    webpage_url = 'https://aqicn.org/city/beijing/dongchengdongsi/'
    dfs = pd.read_html(webpage_url)
    pollutants = dfs[4].iloc[1:7,0:2]
    pollutants = check_float(pollutants)
    
    (wx, wy) = get_wind_vectors(weather_data_current['wind']['deg'], weather_data_current['wind']['speed'])
    
    dt = datetime.fromtimestamp(weather_data_current['dt'])
    
    inputs = {
        'datetime' : dt.replace(second=0, minute=0),
        'PM25': np.float64(dfs[4].iloc[1][1]),
        'PM10' : np.float64(dfs[4].iloc[2][1]),
        'SO2' : np.float64(dfs[4].iloc[5][1]),
        'NO2' : np.float64(dfs[4].iloc[4][1]),
        'CO' : np.float64(dfs[4].iloc[6][1]),
        'O3' : np.float64(dfs[4].iloc[3][1]),
        'TEMP' : K_to_C(weather_data_current['main']['temp']), 
        'PRES' : np.float64(weather_data_current['main']['pressure']),
        'DEWP' : K_to_C(weather_data_history['current']['dew_point']),
        'RAIN' : np.float64(0.0),
        'sin time' : np.sin(2*np.pi*weather_data_current['dt']/(24*60*60)),
        'cos time' :np.cos(2*np.pi*weather_data_current['dt']/(24*60*60)),
        'wx' : wx,
        'wy' : wy
        }
    

    # check if there is any rain accumulation
    if 'rain' in weather_data_current.keys():
        inputs['RAIN'] = weather_data_current['rain']['1h']

    return pd.Series(inputs), pd.Series(inputs).to_json()


# def add_inputs_to_weather_db(dbConnection, data):
    
#     cursor = dbConnection.cursor()
    
#     sql = ("INSERT INTO weather"
#             '(datetime, "PM25", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "sin_time", "cos_time", "wx", "wy")'
#             "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
#             "ON CONFLICT (datetime) DO NOTHING;")
    
#     cursor.execute(sql, data)
    
#     dbConnection.commit()
    
# def add_inputs_to_prediction_db(dbConnection, prediction):
    
#     cursor = dbConnection.cursor()
    
#     sql = ("INSERT INTO predictions"
#             '(datetime, "PM25")'
#             "VALUES (%s, %s)"
#             "ON CONFLICT (datetime) DO NOTHING;")
    
#     cursor.execute(sql, prediction)
    
#     dbConnection.commit()

def check_float(df: pd.DataFrame) -> pd.DataFrame:
    
    #TODO: figure out a nicer way to deal with missing values
    for i,r in df.iterrows():
        try:
            np.float64(r[1])
        except:
            r[1] = -9999
    return df

# def get_history(dbConnection):
#     history = pd.read_sql_query(("SELECT * FROM weather ORDER BY datetime ASC LIMIT 72 OFFSET (SELECT COUNT(*) FROM weather) - 72;"), dbConnection)
#     history.set_index('datetime',inplace=True)
#     return np.array(history).reshape(1,72,14)
    
def predict_future(model, scaler, previous_three_days: np.array) -> np.float32:

    prediction = []
    predictions = model.predict(previous_three_days)
    prediction_copy = predictions[:,23].reshape(-1,1)
    prediction_copies = np.repeat(prediction_copy, previous_three_days.shape[2], axis=-1)
    prediction.append(datetime.fromtimestamp(int(time.time())).replace(second=0,minute=0))
    prediction.append(scaler.inverse_transform(prediction_copies)[0][0])
    return prediction

def print_message(prediction):
    if prediction >= 50 and prediction < 100:
        msg = "The air quality will be generally acceptable."
    elif prediction >= 100 and prediction < 150:
        msg = "Be cautious if you have a respiratory disorder such as asthma."
    elif prediction >= 150 and prediction < 200:
        msg = "Avoid prolonged outdoor exposure."
    elif prediction >= 200 and prediction < 300:
        msg = "The air will be hazardous. Stay indoors if possible."
    elif prediction >= 300:
        msg = "The air will be extremely dangerous."
    else:
        msg = "You can breathe freely!"
    
    return msg

# def get_current_inputs() -> pd.Series:
    
#     current_time = int(time.time())
#     previous_three_days = current_time - 86400*3 # number of seconds in one day = 86400
    
#     API_key = config['API_KEY']
#     latitude = '39.9497'
#     longitude = '116.3891'
#     API_url_history = f'https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={latitude}&lon={longitude}&dt={previous_three_days}&appid={API_key}'
#     API_url_current = f'https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={API_key}'
    
#     # get weather data
#     response_current = requests.get(API_url_current)
#     response_history = requests.get(API_url_history)
#     weather_data_current = json.loads(response_current.text)
#     weather_data_history = json.loads(response_history.text)

#     # scrape page for pollutant info
#     webpage_url = 'https://aqicn.org/city/beijing/dongchengdongsi/'
#     dfs = pd.read_html(webpage_url)
#     pollutants = dfs[4].iloc[1:7,0:2]
#     pollutants = check_float(pollutants)
    
#     (wx, wy) = get_wind_vectors(weather_data_history['current']['wind_deg'], weather_data_history['current']['wind_speed'])
    
#     dt = datetime.fromtimestamp(weather_data_history['current']['dt'])
    
#     inputs = {
#         'datetime' : dt.replace(second=0, minute=0),
#         'PM25': np.float64(dfs[4].iloc[1][1]),
#         'PM10' : np.float64(dfs[4].iloc[2][1]),
#         'SO2' : np.float64(dfs[4].iloc[5][1]),
#         'NO2' : np.float64(dfs[4].iloc[4][1]),
#         'CO' : np.float64(dfs[4].iloc[6][1]),
#         'O3' : np.float64(dfs[4].iloc[3][1]),
#         'TEMP' : K_to_C(weather_data_history['current']['temp']), 
#         'PRES' : np.float64(dfs[4].iloc[8][1]),
#         'DEWP' : K_to_C(weather_data_history['current']['dew_point']),
#         'RAIN' : np.float64(0.0),
#         'sin time' : np.sin(2*np.pi*weather_data_current['dt']/(24*60*60)),
#         'cos time' :np.cos(2*np.pi*weather_data_current['dt']/(24*60*60)),
#         'wx' : wx,
#         'wy' : wy
#         }
#     # check if there is any rain accumulation
#     if 'rain' in weather_data_current.keys():
#         inputs['RAIN'] = weather_data_current['rain']['1h']
#     return pd.Series(inputs), pd.Series(inputs).to_json()

    # prediction = []
    # predictions = model.predict(response)
    # prediction_copy = predictions[:,23].reshape(-1,1)
    # prediction_copies = np.repeat(prediction_copy, response.shape[2], axis=-1)
    # prediction.append(datetime.fromtimestamp(int(time.time())).replace(second=0,minute=0))
    # prediction.append(scaler.inverse_transform(prediction_copies)[0][0])

# EXAMPLE CALLS

#inputs = get_inputs()
#send_request(inputs)
#add_inputs_to_weather_db(cnx,inputs[0].values)
#history = get_history(cnx)

#model = tf.keras.models.load_model('best_model.epoch08-loss0.0114.hdf5')
#scaler = pickle.load(open('scaler.pkl','rb'))
#prediction = predict(model, scaler, history)
#add_inputs_to_prediction_db(cnx, prediction)