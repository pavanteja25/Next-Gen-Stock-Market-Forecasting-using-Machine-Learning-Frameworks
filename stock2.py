import streamlit as st
import pandas as pd 
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
 
import plotly.graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly


st.title ("stock price predictore app")
stocks = st.text_input("enter the stock ID","GOOG")
#Selected_stock = st.selectbox("select dataset for prediction",stocks)
from datetime import date
Start ="2015-01-01"
Today =date.today().strftime("%Y-%m-%d")
n_years=st.slider("Years of prediction:",1,4)
period =n_years*35
def load_data(ticker):
    data =yf.download(ticker,Start,Today)
    data.reset_index(inplace =True)
    return data

data_load_state =st.text("Load data...")
data =load_data(stocks)
data_load_state.text("Loading data....done!")
st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig =go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'], name ='stock_Open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'], name ='stock_Close'))
    fig.layout.update(title_text ="Time series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

df_train =data[['Date','Close']]
df_train =df_train.rename(columns={"Date":"ds","Close":"y"})
m=Prophet ()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)
st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)
