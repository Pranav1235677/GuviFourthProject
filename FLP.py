import pandas as pd
import numpy as np
import streamlit as st
import mlflow
import mlflow.sklearn
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set Streamlit Page Config (Moved to the top)
st.set_page_config(page_title="Flight Price Prediction", page_icon="✈️", layout="wide")

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
if mlflow.active_run():
    mlflow.end_run()

# Cache the dataset loading
@st.cache_data
def load_data():
    df = pd.read_csv("Flight_Price.csv")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')
    df['Dep_Time'] = pd.to_datetime(df['Dep_Time']).dt.hour
    df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time']).dt.hour
    df['Journey_Day'] = df['Date_of_Journey'].dt.day
    df['Journey_Month'] = df['Date_of_Journey'].dt.month
    df['Duration'] = df['Duration'].str.replace('h', '*60').str.replace(' ', '+').str.replace('m', '').apply(eval)
    df['Total_Stops'] = df['Total_Stops'].replace({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}).astype(int)
    df.drop(['Date_of_Journey', 'Route', 'Additional_Info'], axis=1, inplace=True)
    label_enc = LabelEncoder()
    df['Airline'] = label_enc.fit_transform(df['Airline'])
    df['Source'] = label_enc.fit_transform(df['Source'])
    df['Destination'] = label_enc.fit_transform(df['Destination'])
    return df

df = load_data()

# Feature and target split
X = df.drop(columns=['Price'])
y = df['Price']

categorical_features = ['Airline', 'Source', 'Destination']
numerical_features = ['Dep_Time', 'Arrival_Time', 'Duration', 'Total_Stops', 'Journey_Day', 'Journey_Month']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

mlflow.set_experiment("Flight_Price_Prediction")

@st.cache_resource
def train_and_store_models():
    best_model = None
    best_r2 = -np.inf
    best_run_id = None
    trained_models = {}

    for name, model in models.items():
        if mlflow.active_run():
            mlflow.end_run()
        with mlflow.start_run(run_name=name) as run:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mlflow.log_param("model", name)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2_Score", r2)
            mlflow.log_metric("MAE", mae)
            mlflow.sklearn.log_model(pipeline, name)
            joblib.dump(pipeline, f"{name}.pkl")
            trained_models[name] = pipeline
            if r2 > best_r2:
                best_r2 = r2
                best_model = name
                best_run_id = run.info.run_id

    if best_run_id:
        mlflow.register_model(f"runs:/{best_run_id}/{best_model}", "Best_Flight_Price_Model")
    return trained_models

trained_models = train_and_store_models()

def predict_price(model, airline, source, destination, dep_time, arr_time, duration, stops, day, month):
    input_data = pd.DataFrame([[airline, source, destination, dep_time, arr_time, duration, stops, day, month]], columns=categorical_features + numerical_features)
    return model.predict(input_data)[0]

st.title("🚀 Flight Price Prediction")
st.markdown("Predict your flight ticket prices with ease!")
st.sidebar.image("https://www.pngkit.com/png/full/79-799693_airplane-icon-png.png", width=150)

model_choice = st.sidebar.selectbox("✈️ Choose Model", list(models.keys()), index=1)
model = trained_models[model_choice]

st.header("📊 Data Visualization & Insights")
fig, ax = plt.subplots()
sns.boxplot(x=df["Airline"], y=df["Price"], ax=ax)
st.pyplot(fig)

st.sidebar.header("🔹 Flight Details")
airline = st.sidebar.selectbox("Airline", df['Airline'].unique())
source = st.sidebar.selectbox("Source", df['Source'].unique())
destination = st.sidebar.selectbox("Destination", df['Destination'].unique())
dep_time = st.sidebar.number_input("Departure Time (Hour)", min_value=0, max_value=23)
arr_time = st.sidebar.number_input("Arrival Time (Hour)", min_value=0, max_value=23)
duration = st.sidebar.number_input("Duration (Minutes)", min_value=30)
stops = st.sidebar.number_input("Total Stops", min_value=0, max_value=4)
day = st.sidebar.number_input("Journey Day", min_value=1, max_value=31)
month = st.sidebar.number_input("Journey Month", min_value=1, max_value=12)

if st.sidebar.button("🔍 Predict Price"):
    price = predict_price(model, airline, source, destination, dep_time, arr_time, duration, stops, day, month)
    st.sidebar.success(f"Predicted Flight Price: ₹{round(price, 2)}")