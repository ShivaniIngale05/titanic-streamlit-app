import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Titanic ML Dashboard", layout="wide")

# -------------------------------
# Load Data
# -------------------------------
train_df = pd.read_csv("Titanic_train.csv")
test_df = pd.read_csv("Titanic_test.csv")

# -------------------------------
# Title Section
# -------------------------------
st.title("🚢 Titanic Survival Analysis Dashboard")
st.markdown("Interactive Data Analysis and Basic Insights from Titanic Dataset")

# -------------------------------
# Key Metrics
# -------------------------------
total_passengers = train_df.shape[0]
survived = train_df["Survived"].sum()
not_survived = total_passengers - survived
survival_rate = round((survived / total_passengers) * 100, 2)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Passengers", total_passengers)
col2.metric("Survived", survived)
col3.metric("Did Not Survive", not_survived)
col4.metric("Survival Rate (%)", survival_rate)

st.divider()

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("🔎 Filter Data")

gender_filter = st.sidebar.selectbox(
    "Select Gender",
    ["All"] + list(train_df["Sex"].unique())
)

pclass_filter = st.sidebar.selectbox(
    "Select Passenger Class",
    ["All"] + list(train_df["Pclass"].unique())
)

filtered_df = train_df.copy()

if gender_filter != "All":
    filtered_df = filtered_df[filtered_df["Sex"] == gender_filter]

if pclass_filter != "All":
    filtered_df = filtered_df[filtered_df["Pclass"] == pclass_filter]

# -------------------------------
# Tabs Section
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 EDA", "🤖 Prediction Info"])

# -------------------------------
# TAB 1 - Overview
# -------------------------------
with tab1:
    st.subheader("Filtered Data Preview")
    st.dataframe(filtered_df.head())

    st.write("Filtered Data Shape:", filtered_df.shape)

# -------------------------------
# TAB 2 - EDA
# -------------------------------
with tab2:

    st.subheader("Survival Distribution")

    fig1, ax1 = plt.subplots()
    train_df["Survived"].value_counts().plot(kind="bar", ax=ax1)
    ax1.set_title("Survival Count (0 = No, 1 = Yes)")
    ax1.set_xlabel("Survived")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    st.subheader("Survival by Gender")

    fig2, ax2 = plt.subplots()
    pd.crosstab(train_df["Sex"], train_df["Survived"]).plot(kind="bar", ax=ax2)
    ax2.set_title("Survival by Gender")
    st.pyplot(fig2)

# -------------------------------
# TAB 3 - Prediction Info
# -------------------------------
with tab3:
    st.subheader("Model Section (Expandable)")
    st.info("You can integrate Logistic Regression or other ML models here.")
    st.write("This section is ready for adding live prediction functionality.")

st.success("Dashboard Loaded Successfully 🚀")
