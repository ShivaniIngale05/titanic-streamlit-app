import streamlit as st
import pandas as pd
import numpy as np

st.title(" Streamlit is working!")
st.write("Hello Shraddha! Your Streamlit app is running successfully.")

st.title("My Data App")

df = pd.read_csv("Titanic_train.csv")
st.dataframe(df.head())

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df)
else:
    st.info("Please upload a CSV file to see the data.")
