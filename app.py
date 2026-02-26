import streamlit as st
import pandas as pd
import numpy as np

st.title("Titanic Survival Analysis App")

st.write(
    "This application performs exploratory data analysis "
    "on the Titanic training and test datasets."
)

# Load datasets
train_df = pd.read_csv("Titanic_train.csv")
test_df = pd.read_csv("Titanic_test.csv")

st.subheader("Training Dataset Preview")
st.dataframe(train_df.head())

st.subheader("Test Dataset Preview")
st.dataframe(test_df.head())

st.subheader("Basic Dataset Information")
st.write("Training Data Shape:", train_df.shape)
st.write("Test Data Shape:", test_df.shape)

st.success("Application loaded successfully!")

st.subheader("Survival Count")
st.bar_chart(train_df["Survived"].value_counts())
