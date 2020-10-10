import streamlit as st
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
import os
from sqlalchemy import create_engine
import altair as alt

# get data
DATABASE_URL_PSYCOPG2 = os.environ['DATABASE_URL_PSYCOPG2']
con = create_engine(DATABASE_URL_PSYCOPG2).connect()

df = pd.read_sql_table('predictions', con)

st.title('Stock Predictor')

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

# TODO: Check that start_date > end_date is not true
today = datetime.date.today()
start_date = st.sidebar.slider(
  "Start date",
  min_value=today,
  max_value=(today + timedelta(days=365)),
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

selected_tickers = st.sidebar.multiselect('Choose stocks', tickers)

#tickers_df = pd.DataFrame(selected_tickers_table, columns=(['Stock name', 'Market price']))
#tickers_df.set_index('Stock name', inplace=True) # Remove index column
#df.set_index('stock', inplace=True)
#st.table(df)

# graph
#ticker_graph = None
#if len(selected_tickers) > 0:
#  ticker_graph = np.random.randn(20, len(selected_tickers))
#chart_data = pd.DataFrame(ticker_graph, columns=selected_tickers)
chart_data = df[df["stock"].isin(selected_tickers)]
chart = alt.Chart(chart_data).mark_line().encode(
    x='date:T',
    y='price:Q',
    color='stock:N'        
)

#chart_data.set_index('date', inplace=True)
st.altair_chart(chart)