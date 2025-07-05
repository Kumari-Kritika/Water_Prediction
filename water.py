'''import pandas as pd
import numpy as np
import joblib
import pickle
import streamlit as st

# Load the model and structure
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")
 
# Let's create an User interface
st.title("Water Pollutants Predictor")
st.write("Predict the water pollutants based on Year and Station ID")

# User inputs
year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2022)
station_id = st.text_input("Enter Station ID", value='1')

# To encode and then predict
if st.button('Predict'):
    if not station_id:
        st.warning('Please enter the station ID')
    else:
        # Prepare the input
        input_df = pd.DataFrame({'year': [year_input], 'id':[station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        # Align with model cols
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        # Predict
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

        st.subheader(f"Predicted pollutant levels for the station '{station_id}' in {year_input}:")
        predicted_values = {}
        for p, val in zip(pollutants, predicted_pollutants):
            st.write(f'{p}:{val:.2f}')'''






























#ADDED CSS FOR STYLE
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import re

# ------------------- Load Model --------------------
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# ------------------- Load Data ---------------------
df = pd.read_csv("PB_All_2000_2021.csv", sep=";")
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
df['year'] = df['date'].dt.year

# ------------------- Sort Station IDs ---------------------
def sort_station_ids(ids):
    def alphanum_key(key):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', key)]
    return sorted(ids, key=alphanum_key)

available_stations = sort_station_ids(df['id'].dropna().astype(str).unique().tolist())

# ------------------- Streamlit UI ---------------------
st.markdown("<h1 style='text-align: center;'>💧 Water Pollutants Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Predict & visualize water quality trends</h4>", unsafe_allow_html=True)

# ------------------- Input Fields ---------------------
year_input = st.number_input("📅 Select Year", min_value=2000, max_value=2021, value=2021)
station_id = st.selectbox("🏢 Select Station ID", available_stations)

# ------------------- Prediction & Visualization ---------------------
if st.button("🔍 Predict"):
    input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
    input_encoded = pd.get_dummies(input_df, columns=['id'])

    # Align input with training columns
    for col in model_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_cols]

    # Predict pollutants
    prediction = model.predict(input_encoded)[0]
    pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

    st.subheader(f"📈 Predicted pollutant levels for Station `{station_id}` in {year_input}:")
    for p, val in zip(pollutants, prediction):
        st.markdown(f"<div style='font-size:18px; color:#27ae60;'><strong>{p}</strong>: {val:.2f}</div>", unsafe_allow_html=True)

    # ------------------- Plot Trends ---------------------
    st.markdown("### 📊 Historical Trends at this Station")
    station_data = df[df['id'].astype(str) == station_id]
    if not station_data.empty:
        trends = station_data.groupby('year')[pollutants].mean().reset_index()
        st.line_chart(trends.set_index('year'))
    else:
        st.info("No historical data available for this station.")



















