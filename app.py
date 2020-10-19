import streamlit as st
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import os
from sqlalchemy import create_engine
import altair as alt
# import matplotlib.pyplot as plt
from yahoo_fin.stock_info import get_data

def getStockData(index, startDate, endDate):
    data = get_data(index, start_date=startDate, end_date=endDate, index_as_date = True, interval="1d").iloc[:,[3]]
    #checking for nan values
    for i in range(0, len(data)-1):
        if np.isnan(data['close'][i]) == bool(1):
            data['close'][i] = np.mean([data['close'][i-1], data['close'][i+1]])
    return data

# This function's content is only run once per streamlit session and cached.
@st.cache
def getPredictions():
    con = create_engine(os.environ['DATABASE_URL']).connect()
    print('connection created')
    df = pd.read_sql_table('predictions', con)
    con.close()
    print('connection closed')
    return df

df = getPredictions()

st.title('Stock Predictor')

st.text('Stock Predictor tool gives you possibility to compare future prices \nfor different stocks in OMX Helsinki market.')

st.text('Start by choosing your preferred starting point for the historical \nstock data and then choose for how long you wish the stock price \nto be predicted.')

st.text('(Due to environment restrictions new prediction is calculated for \neach stock every third day)')

st.write('Try it out and get **RICH**! :moneybag::moneybag::moneybag:')

# set time values
today = datetime.date.today().strftime("%m/%d/%Y")
minus3years = datetime.date.today() - relativedelta(years=3)
minus_almost3years = minus3years + relativedelta(days=30)
minus3years = minus3years.strftime("%m/%d/%Y")
minus_almost3years = minus_almost3years.strftime("%m/%d/%Y")

# Style
st.markdown(
    """
<style>
.sidebar .sidebar-content {
  padding: 45px;
}
.streamlit-table {
  width: 400px;
}
</style>
""",
  unsafe_allow_html=True,
)

today = datetime.date.today()
start_date = st.sidebar.slider(
  "Start date",
  min_value=(today - relativedelta(years=3) + relativedelta(days=30)),
  max_value=today - relativedelta(days=1),
  format="DD.MM.YYYY"
)

end_date = st.sidebar.slider(
  "End date",
  value=(today + timedelta(days=100)),
  min_value=today,
  max_value=(today + timedelta(days=365)),
  format="DD.MM.YYYY"
)

tickers = list(df["stock"].unique())
tickers.sort()

selected_tickers = st.sidebar.multiselect('Choose stocks', tickers)

# graph (first option)
chart_data = df[(df["stock"].isin(selected_tickers))]

# get history data for selected tickers and append them to chart_data
for ticker in selected_tickers:
  ticker_data = getStockData(ticker,minus_almost3years,today)
  ticker_data.reset_index(inplace=True)
  ticker_data = ticker_data.rename(columns={"index": "date", "close":"price"})
  ticker_data["stock"]=ticker
  chart_data = chart_data.append(ticker_data, ignore_index=True)

#add some helping columns for plotting
chart_data["today"]=today
chart_data["max_price"]=chart_data["price"].max()

#filter by start_date and end_date
date_mask = (chart_data['date'].dt.date > start_date) & (chart_data['date'].dt.date <= end_date)
chart_data = chart_data[date_mask]

#create separate df for highlighting the forecasted area
pred_data = chart_data[chart_data["date"].dt.date>=today]

#build charts
base = alt.Chart(chart_data, height=500, width=700)

line = base.mark_line().encode(
    x='date:T',
    y=alt.Y('price:Q', axis=alt.Axis(title='price')),
    color='stock:N'        
)

'''
The forecast made by model is under the lightblue area.
'''

band = alt.Chart(pred_data, height=500, width=700).mark_area(
    opacity=0.5, color='lightblue'
).encode(
    x='date',
    y=alt.Y('max_price', axis=alt.Axis(title=''))
)

st.altair_chart(line + band)

'''The data is gathered from Yahoo Finance API.'''
