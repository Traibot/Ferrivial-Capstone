# import the required libraries 
import pandas as pd
from matplotlib import pyplot as plt
from pandas import to_datetime
from prophet import Prophet
import pandas_datareader.data as pdr
from datetime import date
import yfinance as yf
import streamlit as st
yf.pdr_override()


# Constants
cost_storage = 10
transport_time = (20, 35)
constant_consumption = 300

# Data
today = date.today()
start_date = '2005-01-01'
symbol = 'BZ=F'

st.title('Forecast of Crude Oil Prices')

# Use pandas_datareader to retrieve the data
df = pdr.get_data_yahoo(symbol, start=start_date, end=today)
df = df.reset_index()
df = df[['Date', 'Close']]
df.columns = ['ds', 'y']
df['ds'] = to_datetime(df['ds'])
m = Prophet()
m.fit(df)

@st.cache_data()
def make_prediction(period):
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    return forecast

forecast_period = st.slider("Forecast period", 9, 90)

forecast = make_prediction(forecast_period)

m.plot(forecast)

from prophet.diagnostics import cross_validation, performance_metrics

# get cached cross validation metrics


@st.cache_data()
def get_cv_metrics():
    df_cv = cross_validation(m, period='45 days', horizon='90 days')
    df_p = performance_metrics(df_cv)
    return df_cv, df_p

df_cv, df_p = get_cv_metrics()

st.write("Mean Absolute Percentage Error (MAPE):", round(df_p[df_p['horizon'] == str(forecast_period) + ' days']['mape'].values[0], 2))

# Plot the forecast
import plotly.graph_objects as go
trace_open = go.Scatter(
x = forecast["ds"],
y = forecast["yhat"],
mode = 'lines',
name="Forecast")

trace_high = go.Scatter(
    x = forecast["ds"],
    y = forecast["yhat_upper"],
    mode = 'lines',
    fill = "tonexty", 
    line = {"color": "#57b8ff"}, 
    name="Higher uncertainty interval"
    )

trace_low = go.Scatter(
    x = forecast["ds"],
    y = forecast["yhat_lower"],
    mode = 'lines',
    fill = "tonexty", 
    line = {"color": "#57b8ff"}, 
    name="Lower uncertainty interval"
    )

trace_close = go.Scatter(
    x = forecast["ds"],
    y = forecast["trend"],
    name="Data values"
    )

data = [trace_open,trace_high,trace_low,trace_close]
layout = go.Layout(title="Bitimen Price Forecast",xaxis_rangeslider_visible=True)
fig = go.Figure(data=data,layout=layout)
st.plotly_chart(fig)


# Create a layout with four columns
col1, col2, col3, col4 = st.columns(4)

# get the first value of the forecast_period
min = forecast['yhat'][-forecast_period:].min()
min_price_30 = forecast['yhat'][-30:].min()
min_price_60 = forecast['yhat'][-60:-30].min()
min_price_90 = forecast['yhat'][-90:-60].min()

# Display the minimum price over the forecast period
with col1:
    st.metric(f"Minimum price over {forecast_period} days", f"${min:.2f}")

# Display the minimum price over the 0-30 days period
with col2:
    if forecast_period < 30:
        st.metric(f"0-30 days", f"$-")
    else:
        min_price = make_prediction(30)
        st.metric(f"0-30 days", f"${min_price_30:.2f}")

# Display the minimum price over the 30-60 days period
with col3:
    if forecast_period < 60:
        st.metric(f"30-60 days", f"$-")
    else:
        min_price = make_prediction(60)
        st.metric(f"30-60 days", f"${min_price_60:.2f}")
with col4:
    if forecast_period < 90:
        st.metric(f"60-90 days", f"$-")
    else:   
        min_price = make_prediction(90)
        st.metric(f"60-90 days", f"${min_price_90:.2f}")


st.subheader('Order quantity')
available_storage = st.slider('Select the amount of available storage (kg)', 450, 1000)
order_quantity = ((2 * constant_consumption * (transport_time[1] + transport_time[0])/2)/60) + available_storage


#The best price, amount to order and best date will be
#def MPCCalculate(df, extraStorage, current_amonut )
from datetime import date, datetime

from datetime import datetime
def MPCCalculate(df, extraStorage, current_amonut ):
    
    minOrder = 300
    storageFee = 0.167 #10euro per kilogram
    avg_transport_day = 30
    today = date.today()
    const_cons = 10  #10kg/day
    
    today = datetime.today().strftime('%Y-%m-%d')
    df['today'] = pd.to_datetime(today)
    df['difference'] = (df['ds'] - df['today']).dt.days
    df['remain_amtOn_delivry'] = current_amonut - (avg_transport_day + df['difference'])*const_cons
    
    #calculating the best price
    df['amtToOrder'] = minOrder + extraStorage - df['remain_amtOn_delivry']
    df['bestPrice'] = df['amtToOrder'] * df['yhat'] * storageFee
    returned = df[df['remain_amtOn_delivry']>0].sort_values(by='bestPrice', ascending=True)
    
    return(returned)


# Slice the DataFrame to extract data between today's date and the date of next week
from datetime import datetime
# Get today's date
today = datetime.today()

mask = (forecast['ds'] >= today)
df_next_week = forecast[mask][['ds', 'yhat']]

import numpy as np
#The best price, amount to order and best date will be
#def MPCCalculate(df, extraStorage, current_amonut )
best = MPCCalculate(df_next_week, order_quantity, 2000 - order_quantity)
best['ds'] = best['ds'].dt.date
#minPrice = best['bestPrice'].min()
bestP = best.nsmallest(1,'bestPrice').reset_index(drop=True)
lowestPrice = np.round(bestP.iloc[0,1],2)
bestDate = bestP.iloc[0,0]
orderAmount = bestP.iloc[0,5]
total_price = bestP.iloc[0,6]
total_price = np.round(total_price,2)
print("the best Date to order is on ", bestP.iloc[0,0], "Recommended Amount to Order: ", orderAmount,  "with best price: ",lowestPrice )
#bestP


# Variables 
from bmdOilPriceFetch import bmdPriceFetch
data_bmd = bmdPriceFetch()
var = (data_bmd['regularMarketPrice']-data_bmd['lastClose'])
var = round(var, 2)
# align the metric display on a row 
col_row1, col_row2, col_row3 = st.columns(3)
# display metrics of the current oil price 


# display metrics of the current storage 



# display the metrics of the order quantity
col_row1.metric('Best Price: ', f"${lowestPrice}")
col_row1.metric('Current oil price: $', data_bmd['regularMarketPrice'], delta = f"{var}%" )
col_row2.metric('Best Date: ', f"{bestDate}")
col_row3.metric('Recommanded Quantity: ', f"{orderAmount} kg")
col_row3.metric('Current storage: ', f"{1000} kg" )

st.subheader(f'The total price: ${total_price}')

# Order button
if st.button('Place order'):
    st.success('Order placed for {} kg'.format(order_quantity))