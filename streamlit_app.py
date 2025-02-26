import streamlit as st
import joblib
import numpy as np

import keras
import tensorflow as tf


model = joblib.load('cancer_risk_model.pkl')


st.title('cancerSIAA')




columns = [
    "Age", 
    "Number of sexual partners", 
    "Smokes (years)", 
    "STDs (number)", 
    "Hormonal Contraceptives (years)", 
    "Num of pregnancies"
]
cu = {
    "Age": 42,  
    "Smokes (years)": 3,
    "Num of pregnancies": 2
}

st.write('Enter the values for all features:')

inputs = []
for col in columns:
    feature_value = st.number_input(f'{col}', value=0.0)
    inputs.append(feature_value)


if None in inputs:
    st.error("One or more input values are missing. Please enter values for all features.")
    st.stop()



cs = False
for feature, threshold in cu.items():
    feature_index = columns.index(feature)  
    if inputs[feature_index] > threshold:
        cs = True


if st.button('Predict'):
    try:
        if cs:
            st.write("Cancer Risk: High Risk ")
        else:
            input_data = np.array(inputs).reshape(1, -1)
            prediction = model.predict(input_data)
            st.write(f'Cancer Risk: {"High Risk" if prediction[0] == 1 else "Low Risk"}')
    except Exception as e:
        st.error(f"Error making prediction: {e}")




