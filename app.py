import streamlit as st
import pandas as pd
import numpy as np

st.title("Titanic Survival Analysis & Prediction App")

st.write("This application performs exploratory data analysis on the Titanic dataset.")

# Load training data
try:
    df = pd.read_csv("Titanic_train.csv")
    st.subheader("Training Data Preview")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("Training dataset not found.")

# Upload new dataset option
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df_uploaded.head())
