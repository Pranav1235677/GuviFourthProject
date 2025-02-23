import pandas as pd
import numpy as np
import streamlit as st
import mlflow
import mlflow.sklearn
import joblib  # For caching models and preprocessing steps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Customer Satisfaction Prediction", page_icon="✈️", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Passenger_Satisfaction.csv")
    return df

df = load_data()

# Data Preprocessing
@st.cache_data
def preprocess_data(df):
    df.fillna(df.median(numeric_only=True), inplace=True)
    categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    ohe = OneHotEncoder(sparse=False, drop='first')
    df_encoded = pd.DataFrame(ohe.fit_transform(df[categorical_cols]))
    df_encoded.columns = ohe.get_feature_names_out(categorical_cols)
    df = df.drop(columns=categorical_cols).join(df_encoded)
    df.columns = df.columns.str.lower()
    X = df.drop(columns=['satisfaction'])
    y = LabelEncoder().fit_transform(df['satisfaction'])
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler, ohe, X.columns

X_train, X_test, y_train, y_test, scaler, ohe, feature_columns = preprocess_data(df)

# MLflow Tracking
mlflow.set_tracking_uri("file:///C:/Users/hustl/OneDrive/Desktop/CustomerSatisfaction/mlruns")
mlflow.set_experiment("Customer_Satisfaction_Prediction")

# Model Training
@st.cache_resource
def train_model():
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100)
    }
    
    best_model, best_acc = None, 0
    
    with mlflow.start_run():
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mlflow.log_metric(f"{name} Accuracy", acc)
            mlflow.log_metric(f"{name} F1-Score", f1)
            
            if acc > best_acc:
                best_acc = acc
                best_model = model
        
        mlflow.sklearn.log_model(best_model, "best_model")
    
    joblib.dump(best_model, "best_model.pkl")
    return best_model

best_model = train_model()

# Streamlit UI

st.title("🌟 Customer Satisfaction Prediction 🌟")

st.header("📊 Exploratory Data Analysis")
st.subheader("Dataset Overview")
st.write(df.describe())

st.subheader("Satisfaction Distribution")
fig, ax = plt.subplots()
sns.countplot(x='satisfaction', data=df, palette='coolwarm', ax=ax)
st.pyplot(fig)

st.subheader("Feature Correlation")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)

# User Input
st.sidebar.header("User Input Features")
user_input = {
    "Gender": st.sidebar.selectbox("Gender", ['Female', 'Male']),
    "Customer Type": st.sidebar.selectbox("Customer Type", ['Loyal Customer', 'Disloyal Customer']),
    "Type of Travel": st.sidebar.selectbox("Type of Travel", ['Personal Travel', 'Business Travel']),
    "Class": st.sidebar.selectbox("Class", ['Business', 'Eco', 'Eco Plus']),
    "Flight distance": st.sidebar.number_input("Flight Distance", min_value=0),
    "Inflight wifi service": st.sidebar.slider("Inflight Wifi Service", 0, 5),
    "Departure Delay in Minutes": st.sidebar.number_input("Departure Delay", min_value=0),
    "Arrival Delay in Minutes": st.sidebar.number_input("Arrival Delay", min_value=0)
}

# Prediction Function
def predict_satisfaction(user_input):
    model = joblib.load("best_model.pkl")
    input_df = pd.DataFrame([user_input])
    input_df_encoded = pd.DataFrame(ohe.transform(input_df[['Gender', 'Customer Type', 'Type of Travel', 'Class']]),
                                    columns=ohe.get_feature_names_out(['Gender', 'Customer Type', 'Type of Travel', 'Class']))
    input_df = input_df.drop(columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'], errors='ignore').join(input_df_encoded)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    input_df = scaler.transform(input_df)
    prediction = model.predict(input_df)[0]
    return "✅ Satisfied" if prediction == 1 else "❌ Dissatisfied"

if st.sidebar.button("Predict Satisfaction"):
    prediction = predict_satisfaction(user_input)
    st.markdown(f"### 🎯 Prediction: {prediction}", unsafe_allow_html=True)

