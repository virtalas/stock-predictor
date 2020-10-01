import streamlit as st
import numpy as np
import pandas as pd
import db
from db import Model

st.title('Stock Predictor')

models = db.fetch_all()
for model in models:
  print(model.name, model.data)

st.write('The script generate_models was last run at', models[0].data)
