import streamlit as st
import numpy as np
import pandas as pd

st.title('Stock Predictor')

f = open("demofile.txt", "r")
st.write('Running script hourly. generate_models.py was last run at', f.read())
f.close()
