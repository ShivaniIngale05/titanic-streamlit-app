import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Titanic ML Dashboard", layout="wide")

# Load Data
train_df = pd.read_csv("Titanic_train.csv")

# Load Model
with open("titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🚢 Titanic Survival Prediction App")

# --------------------------
# Dashboard Metrics
# --------------------------
total = train_df.shape[0]
survived = train_df["Survived"].sum()
rate = round((survived / total) * 100, 2)

col1, col2, col3 = st.columns(3)
col1.metric("Total Passengers", total)
col2.metric("Survived", survived)
col3.metric("Survival Rate (%)", rate)

st.divider()

# --------------------------
# Prediction Section
# --------------------------
st.header("🎯 Live Survival Prediction")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 1, 80, 25)

with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
    parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
    fare = st.number_input("Fare", 0.0, 600.0, 32.0)

if st.button("Predict Survival"):

    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ The passenger is likely to Survive")
    else:
        st.error("❌ The passenger is likely NOT to Survive")
