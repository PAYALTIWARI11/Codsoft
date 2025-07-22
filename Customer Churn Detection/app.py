import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('model.pkl')

st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict churn:")

# Input form
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure (years)", 0, 10, 5)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_credit_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=1000.0, value=50000.0)

# Convert categorical variables
geo_map = {"France": 0, "Germany": 1, "Spain": 2}
gender_map = {"Male": 1, "Female": 0}

# Prediction button
if st.button("Predict"):
    input_data = pd.DataFrame([[
        credit_score, geo_map[geography], gender_map[gender], age,
        tenure, balance, num_of_products, has_credit_card,
        is_active_member, estimated_salary
    ]])
    prediction = model.predict(input_data)
    result = "Customer is likely to churn." if prediction[0] == 1 else "Customer is likely to stay."
    st.success(result)
