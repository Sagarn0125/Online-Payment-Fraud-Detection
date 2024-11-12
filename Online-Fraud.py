import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the title of the app
st.title("Online Fraud Detection Analysis")

# Load dataset
@st.cache
def load_data():
    data = pd.read_csv('onlinefraud.csv')  # Replace with your actual filename
    return data

data = load_data()

# Display the first few rows of the dataset
st.subheader("Dataset Preview")
st.write(data.head())

# Display dataset information
st.subheader("Dataset Information")
buffer = st.empty()
buffer.write(data.info())

# Check for missing values
st.subheader("Missing Values")
missing_values = data.isnull().sum()
st.write(missing_values[missing_values > 0])

# Convert 'step' to integer if it appears as a float
data['step'] = data['step'].astype(int)

# Basic analysis of target variable (isFraud)
st.subheader("Fraudulent Transactions Count")
fraud_counts = data['isFraud'].value_counts()
st.write(fraud_counts)

# Data visualization
st.subheader("Data Visualizations")

# 1. Distribution of transaction types
st.subheader("Transaction Types Distribution")
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='type')
plt.title('Transaction Types')
st.pyplot(plt)

# 2. Amount distribution by transaction type
st.subheader("Transaction Amount Distribution by Type")
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='type', y='amount')
plt.yscale('log')  # Log scale for better visualization
plt.title('Transaction Amount Distribution by Type')
st.pyplot(plt)

# 3. Fraud by transaction type
st.subheader("Fraudulent Transactions by Type")
plt.figure(figsize=(8, 6))
sns.countplot(data=data[data['isFraud'] == 1], x='type')
plt.title('Fraudulent Transactions by Type')
st.pyplot(plt)

# 4. Amount distribution in fraudulent vs. non-fraudulent transactions
st.subheader("Amount Distribution in Fraudulent vs Non-Fraudulent Transactions")
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='isFraud', y='amount')
plt.yscale('log')
plt.title('Amount Distribution in Fraudulent vs Non-Fraudulent Transactions')
st.pyplot(plt)

# Feature Engineering: One-hot encoding for 'type' column
data_encoded = pd.get_dummies(data, columns=['type'], drop_first=True)

# Define feature matrix X and target vector y
X = data_encoded.drop(['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
y = data_encoded['isFraud']

# Display the shapes of X and y
st.subheader("Feature Matrix and Target Vector Shapes")
st.write("Feature matrix shape:", X.shape)
st.write("Target vector shape:", y.shape)

# Footer
st.write("Developed by [Sagar Nainwa]")