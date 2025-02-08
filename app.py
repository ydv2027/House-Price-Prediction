import streamlit as st
import pickle
import numpy as np
import json

# Load the trained model
with open("House_price_predication_model.pickle", "rb") as f:
    model = pickle.load(f)

# Load column names from JSON
with open("columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]

locations = data_columns[3:]  # Exclude first 3 numerical columns (sqft, bath, bhk)

st.title("🏠 House Price Prediction App")

# User input fields
location = st.selectbox("Select Location", locations)
sqft = st.number_input("Enter Square Feet Area", min_value=500, max_value=10000, step=10)
bath = st.number_input("Enter Number of Bathrooms", min_value=1, max_value=10, step=1)
bhk = st.number_input("Enter Number of BHK", min_value=1, max_value=10, step=1)

def predict_price(location, sqft, bath, bhk):
    # Create an array of zeroes matching the trained model's features
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    # Find location index in feature list
    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1  # Set one-hot encoded location

    return model.predict([x])[0]  # Predict using the trained model


if st.button("Predict Price"):
    final_price = predict_price(location, sqft, bath, bhk)
    
    if final_price < 100:
        formatted_price = f"₹ {round(final_price, 2)} lakh"
    else:
        formatted_price = f"₹ {round(final_price / 100, 2)} crore"
    
    st.success(f"🏡 Estimated House Price: {formatted_price}")

