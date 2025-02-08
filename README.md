# Bangalore House Price Prediction

A Machine Learning project to predict house prices in Bangalore using Streamlit.

# Project Overview

This is an end-to-end machine learning project that predicts house prices in Bangalore based on various features such as location, size, and square footage. The project involves data collection, data preprocessing, feature engineering, model selection, and deployment using Streamlit.

![Image Alt](https://github.com/ydv2027/House-Price-Prediction/blob/d704a0b8369fb3de976d5d31dd8bf0009621d405/Website%20Image.png)

# Dataset

The dataset was collected from the Kaggle website.

* Steps Followed

# Importing Libraries

Imported essential libraries: pandas, numpy, and matplotlib for data manipulation and visualization.

# Loading the Dataset

Imported the dataset into a Pandas DataFrame.

Checked for null values and missing data.

# Handling Missing Values

Filled missing values in the bathroom column with the mean value.

Dropped rows with significantly missing or incorrect data.

# Data Cleaning

Dropped irrelevant columns.

Converted the size column into BHK format to ensure data consistency.

Standardized total_sqft values to ensure realistic area calculations.

# Outlier Detection and Removal

Identified outliers where a 3 BHK apartment was priced lower than a 2 BHK in the same area and removed them.

# Feature Engineering

Converted the location column from categorical to numerical using one-hot encoding.

# Model Selection and Training

Selected Ridge Regression as the best machine learning model for price prediction.

Trained the model successfully and achieved accurate predictions.

# Model Serialization

Created a pickle file (model.pkl) to store the trained model.

Saved a JSON file for further reference.

# Deployment

Created an app.py file and requirements.txt for the Streamlit application.

Successfully deployed the project on Streamlit.

The model and application were successfully deployed on Streamlit.

The web app allows users to enter house details and get an estimated price prediction.

# Technologies Used

Python (pandas, numpy, matplotlib, sklearn)

Machine Learning (Ridge Regression)

Streamlit (for web deployment)

Author

* Swatantra Yadav
* Email: swatantra.yadav2027@gmail.com
* LinkedIn: www.linkedin.com/in/contactswatantrayadav2027
* Website: https://house-price-prediction-uffupr2yw9xtcnzasazzvg.streamlit.app/  
