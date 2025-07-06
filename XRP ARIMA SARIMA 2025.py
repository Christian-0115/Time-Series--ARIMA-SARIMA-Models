# Ignore warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import matplotlib.pyplot as plt

def get_xrp_data():
    # Read data
    xrp_data = pd.read_csv('XRP Historical Data.csv')
    
    # Convert Start column to datetime and set it as the index
    xrp_data['Date'] = pd.to_datetime(xrp_data['Date'])
    xrp_data.set_index('Date', inplace = True)
    
    # Reverse index to start from the earliest date
    xrp_data = xrp_data.sort_index(ascending=True)
    
    # Group data by the average close value of each month
    xrp_data = xrp_data.resample('M').mean(numeric_only=True)
    
    xrp_data['Start Date'] = xrp_data.index
    
    return xrp_data

import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox


def adf_test(series):
    """Perform ADF test to determine if a series is stationary"""
    test_results = adfuller(series)
    print('ADF Statistic: ', test_results[0])
    print('P-Value: ', test_results[1])
    print('Critical Values:')
    for thres, adf_stat in test_results[4].items():
        print('\t%s: %.2f' % (thres, adf_stat))
    if test_results[1] < 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")

def adf():
    xrp_data = get_xrp_data()
    
    # Apply Box-Cox transformation and differencing to make the series stationary
    xrp_data['Close Boxcox'], lam = boxcox(xrp_data['Price'])
    xrp_data['Close Stationary'] = xrp_data['Close Boxcox'].diff()
    xrp_data.dropna(inplace=True)
    adf_test(xrp_data['Close Stationary'])
    
    plt.plot(xrp_data['Price'], color ='r')
    plt.plot(xrp_data['Close Boxcox'], color = 'b')
    plt.plot(xrp_data['Close Stationary'], color = 'green')
    
    
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_xrp_autocorr():
    xrp_data = get_xrp_data()
    
    xrp_data['Close Boxcox'], lam = boxcox(xrp_data['Close'])
    xrp_data['Close Stationary'] = xrp_data['Close Boxcox'].diff()
    xrp_data.dropna(inplace=True)
    
    plt.rc("figure", figsize = (8,4))
    plot_acf(xrp_data['Close Stationary'])
    plt.xlabel('Lags', fontsize = 18)
    plt.ylabel('Correlation', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('XRP Autocorrelation Plot', fontsize = 20)
    plt.tight_layout()
    plt.show()
    
def plot_xrp_pacf():
    xrp_data = get_xrp_data()
    
    
    xrp_data['Close Boxcox'], lam = boxcox(xrp_data['Close'])
    xrp_data['Close Stationary'] = xrp_data['Close Boxcox'].diff().diff()
    xrp_data.dropna(inplace=True)
    
    plt.rc("figure", figsize=(11,5))
    plot_pacf(xrp_data['Close Stationary'], method='ywm')
    plt.xlabel('Lags', fontsize=18)
    plt.ylabel('Correlation', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('Partial Autocorrelation Plot', fontsize=20)
    plt.tight_layout()
    plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose


#Multiplicative model
def xrp_mult_dec():
    xrp_data = get_xrp_data()
    xrp_data.rename(columns = {'Price' : 'Price Multiplicative Decomposition'}, inplace = True)
    xrp_mult_decomp_plot = seasonal_decompose(xrp_data['Price Multiplicative Decomposition'], model = "multiplicative")
    xrp_mult_decomp_plot.plot()
    plt.show()

# Additive model
def xrp_add_dec():
    xrp_data = get_xrp_data()
    xrp_data['Close Box-cox'], lam = boxcox(xrp_data['Price'])
    xrp_data.rename(columns = {'Close Box-cox' : 'Price Additive Decomposition'}, inplace = True)
    xrp_add_decomp_plot = seasonal_decompose(xrp_data['Price Additive Decomposition'], model = "additive")
    xrp_add_decomp_plot.plot()
    plt.show()
    
from scipy.special import inv_boxcox
from statsmodels.tsa.arima.model import ARIMA

def xrp_ARIMA_model():
    xrp_data = get_xrp_data()
    
    # boxcox and difference
    xrp_data['Close Boxcox'], lam = boxcox(xrp_data['Price'])
    xrp_data['Close Stationary'] = xrp_data['Close Boxcox'].diff()
    xrp_data.dropna(inplace=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5), dpi=80)
    plot_acf(xrp_data['Close Stationary'], ax = ax1)
    plot_pacf(xrp_data['Close Stationary'], method='ywm', ax = ax2)
    ax1.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    plt.show()
    # from the plots we find p = 13 from pacf and q = 3 from acf
    
    train_set = xrp_data.iloc[:-30]
    test_set = xrp_data.iloc[-30:]
    
    #model
    ARIMA_model = ARIMA(train_set['Close Stationary'], order = (3,1,3)).fit()
    xrp_boxcox_forecasts = ARIMA_model.forecast(len(test_set)+12)
    xrp_forecasts = inv_boxcox(xrp_boxcox_forecasts,lam)
    
    # xrp_forecasts = pd.DataFrame(xrp_forecasts, index = test_set.index, columns=['ARIMA Forecasts'])

    plt.figure(figsize=(14,7))
    plt.plot(train_set['Price'], label= 'Training Data', color = 'blue')
    plt.plot(test_set['Price'], label = 'Test Data', color = 'green')
    plt.plot(xrp_forecasts, label = 'ARIMA Forecast', color = 'red') 
    plt.xlabel('Date')
    plt.ylabel('XRP Close Price')
    plt.title('XRP Price Forecast using ARIMA')
    plt.legend()
    plt.show()
    
def xrp_SARIMA_model():
    xrp_data = get_xrp_data()
    
    # boxcox and difference
    xrp_data['Close Boxcox'], lam = boxcox(xrp_data['Price'])
    xrp_data['Close Stationary'] = xrp_data['Close Boxcox'].diff()
    xrp_data.dropna(inplace=True)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5), dpi=80)
    plot_acf(xrp_data['Close Stationary'], ax = ax1)
    plot_pacf(xrp_data['Close Stationary'], method='ywm', ax = ax2)
    ax1.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    plt.show()
  
    train_set = xrp_data.iloc[:-30]
    test_set = xrp_data.iloc[-30:]
    
    #model
    ARIMA_model = ARIMA(train_set['Close Boxcox'], order = (7,1,3),seasonal_order=(7,1,3,12)).fit()
    xrp_boxcox_forecasts = ARIMA_model.forecast(len(test_set)+60)
    xrp_forecasts = inv_boxcox(xrp_boxcox_forecasts,lam)
    
    plt.figure(figsize=(14,7))
    plt.plot(train_set['Price'], label= 'Training Data', color = 'blue')
    plt.plot(test_set['Price'], label = 'Test Data', color = 'green')
    plt.plot(xrp_forecasts, label = 'SARIMA Forecast', color = 'red') 
    plt.xlabel('Date')
    plt.ylabel('XRP Close Price')
    plt.title('XRP Price Forecast using SARIMA')
    plt.legend()
    plt.show()
    
    #Write the results in a table, want to write the length of the training set
    #since thats the only thing we can compare the variance with (actual vs predicted(test)) 
    #in this case the length of the training set is 30 (refer above)
    
    #build an index for the forecasted values
    test_indx = test_set.index
    future_indx = pd.date_range(start=test_indx[-1] + pd.Timedelta(days=1),periods = 60, freq = 'M')
    fc_index  = test_indx.union(future_indx)
 
    #build a data frame
    combined_df = pd.DataFrame({
        'Actual Price' : pd.concat([test_set['Price'], 
                                   pd.Series([pd.NA]*60, index = future_indx)]),
        'Forecast Price' : pd.Series(xrp_forecasts.values, index = fc_index)
        })
    
    #calculate the variances between actual and predicted values
    combined_df['Variance'] = combined_df['Forecast Price'] - combined_df['Actual Price']
    combined_df['Absolute Variance'] = combined_df['Variance'].abs()
    combined_df['% Variance'] = combined_df['Variance'] / combined_df['Actual Price'] * 100
    
    #display the table
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(combined_df)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    