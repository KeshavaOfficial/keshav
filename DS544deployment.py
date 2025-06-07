import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
import calendar
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping as KerasEarlyStopping
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from collections import deque

st.set_page_config(page_title="PJMW Forecasting App", layout="wide")
st.title("⚡ PJMW Electricity Demand Forecasting")
st.sidebar.title("📂 Navigation")
section = st.sidebar.radio("Go to", ["EDA Dashboard", "Modeling & Forecasting", "Forecast Comparison"])

@st.cache_data

def load_data():
    df = pd.read_csv("PJMW_hourly.csv")
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.drop_duplicates(subset='Datetime')
    df = df.sort_values('Datetime')
    df.set_index('Datetime', inplace=True)
    df = df.asfreq('h')
    df['MW_clean'] = df['PJMW_MW'].interpolate(method='time')
    return df

if section == "EDA Dashboard":
    df = load_data()
    st.header("📊 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Time Series", "Distributions", "Seasonality & Clusters"])

    with tab1:
        st.subheader("Electricity Demand Over Time")
        fig1 = px.line(df, x=df.index, y='MW_clean', title='Hourly Electricity Demand')
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("7-Day Rolling Average")
        df['rolling'] = df['MW_clean'].rolling(168).mean()
        fig2 = px.line(df, x=df.index, y=['MW_clean', 'rolling'], labels={"value": "MW", "variable": "Legend"}, title='Demand & Rolling Average')
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Distribution of Demand")
        fig3, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df['MW_clean'], kde=True, ax=ax, bins=30, color='skyblue')
        st.pyplot(fig3)

        st.subheader("Boxplot")
        fig4, ax = plt.subplots(figsize=(8, 2))
        sns.boxplot(x=df['MW_clean'], ax=ax, color='salmon')
        st.pyplot(fig4)

    with tab3:
        st.subheader("Weekly Pattern (Day of Week vs Demand)")
        df['dayofweek'] = df.index.dayofweek
        weekly_avg = df.groupby('dayofweek')['MW_clean'].mean()
        fig5 = px.bar(x=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], y=weekly_avg.values, title='Avg MW by Day of Week')
        st.plotly_chart(fig5, use_container_width=True)

        st.subheader("Hourly Pattern")
        df['hour'] = df.index.hour
        hourly_avg = df.groupby('hour')['MW_clean'].mean()
        fig6 = px.bar(x=hourly_avg.index, y=hourly_avg.values, title='Avg MW by Hour')
        st.plotly_chart(fig6, use_container_width=True)

        st.subheader("Heatmap: Day vs Hour")
        pivot = df.pivot_table(values='MW_clean', index='dayofweek', columns='hour', aggfunc='mean')
        fig7, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(pivot, cmap='YlGnBu', ax=ax)
        ax.set_xlabel("Hour"); ax.set_ylabel("Day of Week")
        st.pyplot(fig7)

        st.subheader("Clustered Daily Patterns")
        daily_hourly = df['MW_clean'].groupby([df.index.date, df.index.hour]).mean().unstack()
        daily_hourly_clean = daily_hourly.dropna()
        kmeans = KMeans(n_clusters=3, random_state=42)
        daily_hourly_clean['cluster'] = kmeans.fit_predict(daily_hourly_clean)

        fig8, ax = plt.subplots(figsize=(10, 4))
        for c in range(3):
            avg_pattern = daily_hourly_clean[daily_hourly_clean['cluster'] == c].drop('cluster', axis=1).mean()
            ax.plot(avg_pattern, label=f'Cluster {c}')
        ax.set_title("Clustered Daily Demand Patterns")
        ax.set_xlabel("Hour of Day"); ax.set_ylabel("MW")
        ax.legend(); st.pyplot(fig8)

if section == "Modeling & Forecasting":
    st.header("🧠 Forecasting Models")
    df = load_data()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    for lag in [1, 24, 168]:
        df[f'lag_{lag}'] = df['PJMW_MW'].shift(lag)
    df['rolling_mean_24'] = df['PJMW_MW'].rolling(24).mean()
    df['rolling_std_24'] = df['PJMW_MW'].rolling(24).std()
    df.dropna(inplace=True)

    FEATURES = ["lag_1", "lag_24", "lag_168", "hour", "dayofweek", "is_weekend", "rolling_mean_24", "rolling_std_24"]
    X = df[FEATURES]
    y = df["PJMW_MW"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=168)

    st.subheader("📌 XGBoost Forecast")
    xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, objective="reg:squarederror")
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    st.write("**XGBoost Performance:**")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
    st.write(f"R²: {r2_score(y_test, y_pred):.2f}")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(y_test.index, y_test, label="Actual")
    ax.plot(y_test.index, y_pred, label="Predicted")
    ax.set_title("XGBoost Forecast – Last Week")
    ax.legend(); st.pyplot(fig)

    st.subheader("🔮 Recursive 7-day Forecast")
    lags = deque(df["PJMW_MW"].iloc[-168:], maxlen=168)
    future_preds = []
    for step in range(1, 169):
        ts = df.index[-1] + pd.Timedelta(hours=step)
        X_now = np.array([[lags[0], lags[23], lags[-1], ts.hour, ts.dayofweek, int(ts.dayofweek in [5, 6]), np.mean(lags), np.std(lags) or 0]])
        pred = xgb_model.predict(X_now)[0]
        future_preds.append(pred)
        lags.append(pred)
    future_index = pd.date_range(start=df.index[-1] + pd.Timedelta(hours=1), periods=168, freq="h")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df.index[-168:], df["PJMW_MW"].iloc[-168:], label="Past Week")
    ax2.plot(future_index, future_preds, label="XGBoost Forecast")
    ax2.legend(); st.pyplot(fig2)

st.sidebar.markdown("---")
st.sidebar.info("Created with ❤️ using Streamlit")
