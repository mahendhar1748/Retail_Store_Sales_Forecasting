# pip install streamlit prophet yfinance plotly

'''Prophet is a procedure for forecasting time series data based on an additive model 
where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.'''

import pandas as pd
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go



# Giving the title For Application
st.title('Retail Store Forecast App')

#Choosing How many months you have to Predict
n_months = st.slider('No of Months To Predict Sales:', 1, 12)

#Choosing How many days you have to Predict
#n_days = st.slider('Years of prediction:', 1, 30)

# Here we are predicting for 30 days ---> we can cange if we want
#period = n_days
period = n_months * 30


def load_data(ticker):
    data = pd.read_csv('Final_data.CSV')
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data('Final_data.CSV')
data_load_state.text("Loading data ... done")

st.subheader("Raw Data")
st.write(data.tail())
# Check for missing values
st.write(data.isna().sum())

# Handle missing values (e.g., drop rows with NaN values)
data = data.dropna()

# Ensure the DataFrame has at least two non-NaN rows
if data.shape[0] < 2:
    st.error("Dataframe has less than 2 non-NaN rows")
else:
    st.write("Dataframe is ready for forecasting")
    # Proceed with forecasting logic


# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Revenue'], name="Revenue"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Revenue']]
df_train = df_train.rename(columns={"Date": "ds", "Revenue": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_months} months')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
