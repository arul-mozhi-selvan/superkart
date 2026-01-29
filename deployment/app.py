import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

# Load the trained model from huggingface hub
model = xgb.XGBClassifier(enable_categorical=True)
model_path = hf_hub_download(repo_id="arulmozhiselvan/arul-gl-tourism-xgboost-model", filename="best_model.json")
model.load_model(model_path)

st.title("Tourism Product Purchase Prediction")
st.write("Enter the details to predict if a customer will purchase the tourism product.")
def user_input_features():
    # get user inputs for all features from numerical and categorical columns
    inputs = {}
    for col in ['Age', 'DurationOfPitch', 'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome']:
        value = st.number_input(f"Enter {col}:", value=0)
        inputs[col] = value
    for col in ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'ProductPitched', 'CityTier', 'Designation', 'MonthlyIncome', 'Country']:
        value = st.text_input(f"Enter {col}:")
        inputs[col] = value
    features = pd.DataFrame(inputs, index=[0])
    for col in ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'ProductPitched', 'CityTier', 'Designation', 'MonthlyIncome', 'Country']:
        features[col] = features[col].astype("category")
    return features

input_df = user_input_features()
# arrange the input dataframe columns in the same order as training data


st.subheader("User Input Features")

st.write(input_df)
#one hot encode categorical variables
input_df = pd.get_dummies(input_df, drop_first=True)
input_df = input_df[['Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome', 'TypeofContact_Self Enquiry', 'Occupation_Large Business', 'Occupation_Salaried', 'Occupation_Small Business', 'Gender_Female', 'Gender_Male', 'ProductPitched_Deluxe', 'ProductPitched_King', 'ProductPitched_Standard', 'ProductPitched_Super Deluxe', 'MaritalStatus_Married', 'MaritalStatus_Single', 'MaritalStatus_Unmarried', 'Designation_Executive', 'Designation_Manager', 'Designation_Senior Manager', 'Designation_VP']]
# Predict the output
prediction = model.predict(input_df)
st.subheader("Prediction")
st.write("The customer will purchase the product." if prediction[0] == 1 else "The customer will not purchase the product.")
