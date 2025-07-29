import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

#Now to Load trained model and encoders
model = pk.load(open('model.pkl', 'rb'))


with open('encoders.pkl', 'rb') as f:
    encoders = pk.load(f)  

st.header('Car Price Prediction ML Model')

# Load dataset for UI option 
dataset = pd.read_csv('C:/Users/abhia/OneDrive/Desktop/cardataset.csv')


def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

dataset['name'] = dataset['name'].apply(get_brand_name)

# For taking the inputs from the user
name = st.selectbox('Select Car Brand', sorted(dataset['name'].unique()))
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel Type', sorted(dataset['fuel'].unique()))
seller_type = st.selectbox('Seller Type', sorted(dataset['seller_type'].unique()))
transmission = st.selectbox('Transmission Type', sorted(dataset['transmission'].unique()))
owner = st.selectbox("Ownership", sorted(dataset['owner'].unique()))
mileage = st.slider('Car Mileage (km/l)', 10.0, 40.0)
engine = st.slider('Engine Capacity (cc)', 700, 5000)
max_power = st.slider('Maximum Power (bhp)', 0.0, 200.0)
seats = st.slider('No of Seats', 2, 10)

if st.button('Predict'):
    try:
        name_encoded = encoders['name'].transform([name])[0]
        fuel_encoded = encoders['fuel'].transform([fuel])[0]
        seller_type_encoded = encoders['seller_type'].transform([seller_type])[0]
        transmission_encoded = encoders['transmission'].transform([transmission])[0]
        owner_encoded = encoders['owner'].transform([owner])[0]
    except Exception as e:
        st.error(f"Encoding Error: {e}")
    else:
        input_data = pd.DataFrame(
            [[name_encoded, year, km_driven, fuel_encoded, seller_type_encoded,
              transmission_encoded, owner_encoded, mileage, engine, max_power, seats]],
            columns=['name','year','km_driven','fuel','seller_type','transmission',
                     'owner','mileage','engine','max_power','seats']
        )

        # Now Predicting the price using the model
        prediction = model.predict(input_data)
        st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")

