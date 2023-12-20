import tensorflow as tf
import numpy as np
import pandas as pd
import pyodbc
from tensorflow.keras.utils import get_file
from fastapi import HTTPException
import os

storage_uri = os.getenv('STORAGE_URI')
connection_string = os.getenv('DATABASE_URI')
connection = pyodbc.connect(connection_string)
cursor = connection.cursor()

def get_commodities():
    try:
        get_commodities_query = "SELECT Id, Name FROM Commodities"
        cursor.execute(get_commodities_query)
        columns = [column[0] for column in cursor.description]
        commodities = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return commodities
    except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

def get_dataset(commodity_id):
    try:
        get_dataset_query = "SELECT pd.DatasetLink FROM PredictionDatasets pd WHERE pd.IsUsed = 1 AND pd.CommodityId = ?"
        cursor.execute(get_dataset_query, commodity_id)
        get_dataset_results = cursor.fetchall()
        return get_dataset_results[0].DatasetLink
    except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
def get_model(commodity_id):
    try:
        get_model_query = "SELECT pm.ModelLink FROM PredictionModels pm WHERE pm.IsUsed = 1 AND pm.CommodityId = ?"
        cursor.execute(get_model_query, commodity_id)
        get_model_results = cursor.fetchall()
        return get_model_results[0].ModelLink
    except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

def util(dataset_name, model_name, num_prediction):    

    # ======================================================================= #
    # Nama file dataset
    ds_komoditas = dataset_name

    df = pd.read_csv(f'datasets/{ds_komoditas}.csv')
    df['Date']=pd.to_datetime(df['Date'], format='%Y-%m-%d')
    last_date = df['Date'].iloc[-1]
    # set the Date column be the index of our dataset
    df.set_index('Date', inplace=True)

    temps = df['Price'].values.astype('float32')
    time_step = df.index.values

    prices = np.array(temps)


    # ======================================================================= #
    def normalize_dataset(prices):
        min = np.min(prices)
        max = np.max(prices)
        prices = prices - min
        prices = prices / (max - min)
        return prices

    max = np.max(temps)
    min = np.min(temps)

    time = np.array(time_step)
    prices = normalize_dataset(temps)


    # ======================================================================= #
    # Split Dataset
    split_time = int(len(prices) * 0.7)
    # Hyperparameter
    window_size = 60
    batch_size = 32
    shuffle_buffer_size = 1000


    # ======================================================================= #
    # Load model
    model = tf.keras.models.load_model(f'models/{model_name}.h5')

    # ======================================================================= #
    # Predict
    #Number of future predictions
    future_forecast = num_prediction

    #Use the last window_size data points for the initial sequence
    sequence = prices[-window_size:]

    #Store the predictions
    predictions = []

    #Loop to predict future dates
    for i in range(future_forecast):
        # Reshape the sequence
        sequence_reshaped = sequence[-window_size:].reshape(1, window_size)

    #Predict the next data point
        predicted_value = model.predict(sequence_reshaped, verbose=0)[0]

    #Append the predicted value to the sequence
        sequence = np.append(sequence, predicted_value)

    #Append the predicted value to the predictions
        predictions.append(predicted_value)

    predictions = (np.array(predictions) * (max-min) + min).flatten()


    # ======================================================================= #
    # Response
    # Generate date range starting from the day after the last date
    date_range = pd.date_range(last_date, periods=len(predictions) + 1, freq='D')[1:]

    # Combine dates and numbers into a dictionary
    data = {str(date.date()): str(number) for date, number in zip(date_range, predictions)}

    return data

def util_dua(commodity_id, num_prediction):    

    # ======================================================================= #
    # Nama file dataset
    used_dataset = get_dataset(commodity_id)
    used_model = get_model(commodity_id)
    # ds_komoditas = dataset_name
    print(used_dataset)

    df = pd.read_csv(f'{storage_uri}{used_dataset}')
    df['Date']=pd.to_datetime(df['Date'], format='%Y-%m-%d')
    last_date = df['Date'].iloc[-1]
    # set the Date column be the index of our dataset
    df.set_index('Date', inplace=True)

    temps = df['Price'].values.astype('float32')
    time_step = df.index.values

    prices = np.array(temps)


    # ======================================================================= #
    def normalize_dataset(prices):
        min = np.min(prices)
        max = np.max(prices)
        prices = prices - min
        prices = prices / (max - min)
        return prices

    max = np.max(temps)
    min = np.min(temps)

    time = np.array(time_step)
    prices = normalize_dataset(temps)


    # ======================================================================= #
    # Split Dataset
    split_time = int(len(prices) * 0.7)
    # Hyperparameter
    window_size = 60
    batch_size = 32
    shuffle_buffer_size = 1000


    # ======================================================================= #
    # Load model
    model_url = f'{storage_uri}{used_model}'
    model_path = get_file(f'{commodity_id}.h5', origin=model_url, verify=False)
    model = tf.keras.models.load_model(model_path)

    # ======================================================================= #
    # Predict
    #Number of future predictions
    future_forecast = num_prediction

    #Use the last window_size data points for the initial sequence
    sequence = prices[-window_size:]

    #Store the predictions
    predictions = []

    #Loop to predict future dates
    for i in range(future_forecast):
        # Reshape the sequence
        sequence_reshaped = sequence[-window_size:].reshape(1, window_size)

    #Predict the next data point
        predicted_value = model.predict(sequence_reshaped, verbose=0)[0]

    #Append the predicted value to the sequence
        sequence = np.append(sequence, predicted_value)

    #Append the predicted value to the predictions
        predictions.append(predicted_value)

    predictions = (np.array(predictions) * (max-min) + min).flatten()


    # ======================================================================= #
    # Response
    # Generate date range starting from the day after the last date
    date_range = pd.date_range(last_date, periods=len(predictions) + 1, freq='D')[1:]

    # Combine dates and numbers into a dictionary
    data = {str(date.date()): str(number) for date, number in zip(date_range, predictions)}

    return data