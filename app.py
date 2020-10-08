import streamlit as st
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta

import db
from db import Model

# Example code for database:
models = db.fetch_all()
for model in models:
  print(model.name, model.data)
# st.write('The script generate_models was last run at', models[0].data)

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

# Mock tickers
tickers = ['AAPL', 'FDSEG', 'SDFG', 'SDFE', 'GFDG']

selected_tickers = st.sidebar.multiselect('Choose stocks', tickers)

# Mock market price table
selected_tickers_table = []
for ticker in selected_tickers:
  selected_tickers_table.append([ticker, '2.54'])

tickers_df = pd.DataFrame(selected_tickers_table, columns=(['Stock name', 'Market price']))
tickers_df.set_index('Stock name', inplace=True) # Remove index column
st.table(tickers_df)

# Mock graph
ticker_graph = None
if len(selected_tickers) > 0:
  ticker_graph = np.random.randn(20, len(selected_tickers))
chart_data = pd.DataFrame(ticker_graph, columns=selected_tickers)
st.line_chart(chart_data)
