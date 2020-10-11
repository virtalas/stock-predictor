#importing required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
#to plot within notebook
import matplotlib.pyplot as plt
from datetime import date
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine
import psycopg2 
import os


# function to calculate percentage difference considering baseValue as 100%
def percentageChange(baseValue, currentValue):
    return((float(currentValue)-baseValue) / abs(baseValue)) *100.00

# function to get the actual value using baseValue and percentage
def reversePercentageChange(baseValue, percentage):
    return float(baseValue) + float(baseValue * percentage / 100.00)

# function to transform a list of values into the list of percentages. For calculating percentages for each element in the list
# the base is always the previous element in the list.
def transformToPercentageChange(x):
    baseValue = x[0]
    x[0] = 0
    for i in range(1,len(x)):
        pChange = percentageChange(baseValue,x[i])
        baseValue = x[i]
        x[i] = pChange

# function to transform a list of percentages to the list of actual values. For calculating actual values for each element in the list
# the base is always the previous calculated element in the list.
def reverseTransformToPercentageChange(baseValue, x):
    x_transform = []
    for i in range(0,len(x)):
        value = reversePercentageChange(baseValue,x[i])
        baseValue = value
        x_transform.append(value)
    return x_transform

#read the data file

# store the first element in the series as the base value for future use.


# create a new dataframe which is then transformed into relative percentages
from yahoo_fin.stock_info import get_data

def getStockData(index, startDate, endDate):
    data = get_data(index, start_date=startDate, end_date=endDate, index_as_date = True, interval="1d").iloc[:,[3]]
    #checking for nan values
    for i in range(0, len(data)-1):
        if np.isnan(data['close'][i]) == bool(1):
            data['close'][i] = np.mean([data['close'][i-1], data['close'][i+1]])
    return data

def getTrainData(data):

    baseValue = data['close'][0]
    new_data = data
           
    # transform the 'Close' series into relative percentages
    transformToPercentageChange(new_data['close'])

    # create train and test sets
    
    dataset = new_data.values
    for i in range(0, len(dataset)-1):
        if np.isnan(dataset[i]) == bool(1):
            dataset[i] = np.mean([dataset[i-1], dataset[i+1]])

#train, valid = train_test_split(dataset, train_size=0.99, test_size=0.01, shuffle=False)
    train = dataset

# convert dataset into x_train and y_train.
# prediction_window_size is the size of days windows which will be considered for predicting a future value.
    prediction_window_size = 60
    x_train, y_train = [], []
    for i in range(prediction_window_size,len(train)):
        x_train.append(dataset[i-prediction_window_size:i,0])
        y_train.append(dataset[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    
    return x_train, y_train, baseValue, new_data

##################################################################################################
# create and fit the LSTM network
# Initialising the RNN
def createModel(x_train):
    model = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    
    # Adding the output layer
    model.add(Dense(units = 1))
    # Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    # Fitting the RNN to the Training set
    return model
    

##################################################################################################

#predicting future values, using past 60 from the train data
# for next 10 yrs total_prediction_days is set to 3650 days
def prediction(data, baseValue, predictionPeriod):
    model.fit(x_train, y_train, epochs = 100, batch_size = 32)
    total_prediction_days = 4000
    inputs = data[-total_prediction_days:].values
    inputs = inputs.reshape(-1,1)
    
    # create future predict list which is a two dimensional list of values.
    # the first dimension is the total number of future days
    # the second dimension is the list of values of prediction_window_size size
    X_predict = []
    for i in range(60,inputs.shape[0]):
        X_predict.append(inputs[i-60:i,0])
    X_predict = np.array(X_predict)
    
    # predict the future
    X_predict = np.reshape(X_predict, (X_predict.shape[0],X_predict.shape[1],1))
    future_closing_price = model.predict(X_predict)
    
    train = data
    date_index = pd.to_datetime(train.index)
    
    # #converting dates into number of days as dates cannot be passed directly to any regression model
    x_days = (date_index - pd.to_datetime('1970-01-01')).days
    
    # # we are doing prediction for next 5 years hence prediction_for_days is set to 1500 days.
    prediction_for_days = predictionPeriod
    future_closing_price = future_closing_price[:prediction_for_days]
    
    # create a data index for future dates
    x_predict_future_dates = np.asarray(pd.RangeIndex(start=x_days[-1] + 1, stop=x_days[-1] + 1 + (len(future_closing_price))))
    future_date_index = pd.to_datetime(x_predict_future_dates, origin='1970-01-01', unit='D')
    
    # transform a list of relative percentages to the actual values
    train_transform = reverseTransformToPercentageChange(baseValue, train['close'])
    
    # for future dates the base value the the value of last element from the training set.
    baseValue = train_transform[-1]
    #valid_transform = reverseTransformToPercentageChange(baseValue, valid['close'])
    future_closing_price_transform = reverseTransformToPercentageChange(baseValue, future_closing_price)

    return future_date_index, future_closing_price_transform


def plotPredictions(data,future_date, future_price):
    # plot the graphs
    plt.figure(figsize=(16,8))
    plt.plot(future_date,future_price, label='Predicted Close')
    plt.plot(data['close'], label='Close Price history')

    # set the title of the graph
    plt.suptitle('Stock Market Predictions', fontsize=16)
    
    # set the title of the graph window
    fig = plt.gcf()
    fig.canvas.set_window_title('Stock Market Predictions')
    
    #display the legends
    plt.legend()
    
    #display the graph
    plt.show()


def getList(future_date, future_price):
#the list of predicted dates from 
    dates = pd.DataFrame(future_date, columns=["date"])
    values = pd.DataFrame(future_price, columns=["price"])
    result = pd.concat([dates, values], axis=1)
    return result

#####################################################################################################
#start running models
#####################################################################################################

#setting the start and end date of the train data
    
today = date.today().strftime("%m/%d/%Y")
minus3years = date.today() - relativedelta(years=3)
minus3years = minus3years.strftime("%m/%d/%Y")

#get all tickers
ticker_df = pd.read_csv("tickers.csv", header=None)
tickers = ticker_df[0].tolist()
tickers.sort()

#db connections 
DATABASE_URL_PSYCOPG2 = os.environ['DATABASE_URL_PSYCOPG2']
engine = create_engine(DATABASE_URL_PSYCOPG2)

DATABASE_URL = os.environ['DATABASE_URL']
connection = psycopg2.connect(DATABASE_URL)
cursor = connection.cursor()

#check first if table exists already
check_table = '''
    SELECT EXISTS (
   SELECT FROM information_schema.tables 
   WHERE  table_name   = 'predictions'
   );
'''
cursor.execute(check_table)
test = cursor.fetchall()
#if table doesn't exist - create it first
if (test[0][0]==False):
    create_table_query = '''
      CREATE TABLE IF NOT EXISTS predictions (
        date DATE NOT NULL,
        price float(2) NOT NULL,
        stock VARCHAR(100) NOT NULL
      );
    '''
    cursor.execute(create_table_query)
#truncate predictions
else:
    cursor.execute("TRUNCATE predictions;")
connection.commit()
connection.close()

#loop through tickers 
for ticker in tickers[0:6]:
    print(ticker)
    #selecting stock index and with a time range
    stockData = getStockData(ticker,minus3years,today) 
    if (stockData.index[0].date().strftime("%m/%d/%Y")==minus3years):
      #preparing train data
      x_train, y_train, baseValue, data = getTrainData(stockData)
      #setting model properties
      model = createModel(x_train)
      #running model and predicting future price based on future date
      future_date, future_price = prediction(data,baseValue,predictionPeriod = 100)
      #plotting the results
      #stockData = getStockData(ticker,minus2years,today)
      #plotPredictions(stockData,future_date, future_price)
      #getting the results in a list
      result = getList(future_date, future_price)
      #add ticker column
      result["stock"]=ticker
      #save results to db
      result.to_sql('predictions', engine, if_exists='append',index=False)