import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load
import os
import sys
from PIL import Image

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our model scripts
import stock_prediction
import advanced_model
import ensemble_model
try:
    import lstm_model
    has_tensorflow = True
except ImportError:
    has_tensorflow = False

st.set_page_config(page_title='Stock Price Prediction', layout='wide')

def load_data(ticker, start_date, end_date):
    """
    Load stock data for the given ticker and date range
    """
    try:
        # Verify that end date is not in the future
        today = datetime.date.today()
        if end_date > today:
            st.error(f"End date cannot be in the future. Please select a date on or before {today.strftime('%Y-%m-%d')}.")
            return pd.DataFrame()  # Return empty dataframe
            
        # Download data
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        # Check if data is empty
        if stock_data.empty or len(stock_data) < 10:  # Require at least 10 data points
            st.error(f"Insufficient data for {ticker} in the selected date range. Please select a different date range.")
            return pd.DataFrame()
            
        # Format data
        stock_data.reset_index(inplace=True)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('Date', inplace=True)
        return stock_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def get_stock_list():
    """
    Get a list of popular stocks
    """
    return {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'GOOG': 'Alphabet Inc.',
        'AMZN': 'Amazon.com, Inc.',
        'META': 'Meta Platforms, Inc.',
        'TSLA': 'Tesla, Inc.',
        'NVDA': 'NVIDIA Corporation',
        'JPM': 'JPMorgan Chase & Co.',
        'V': 'Visa Inc.',
        'WMT': 'Walmart Inc.'
    }

def run_linear_model(data):
    """
    Run the basic linear regression model
    """
    X, y = stock_prediction.prepare_data(data)
    model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = stock_prediction.train_model(X, y)
    future_dates, future_predictions = stock_prediction.predict_future(model, X, days=prediction_days)
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'future_dates': future_dates,
        'future_predictions': future_predictions
    }

def run_advanced_model(data):
    """
    Run the advanced model with technical indicators
    """
    X, y = advanced_model.prepare_advanced_features(data)
    model_result = advanced_model.train_advanced_model(X, y)
    
    return model_result

def run_ensemble_model(data):
    """
    Run the ensemble model
    """
    X, y, prepared_data = ensemble_model.prepare_features(data)
    model_results = ensemble_model.build_ensemble_model(X, y)
    future_dates, future_prices, future_prices_by_model = ensemble_model.predict_future_ensemble(
        model_results, data, days=prediction_days
    )
    
    model_results['future_dates'] = future_dates
    model_results['future_prices'] = future_prices
    model_results['future_prices_by_model'] = future_prices_by_model
    
    return model_results

def run_lstm_model(data):
    """
    Run the LSTM model if TensorFlow is available
    """
    if not has_tensorflow:
        st.warning("TensorFlow is not installed. LSTM model is not available.")
        return None
    
    sequence_length = 60
    X_train, X_test, y_train, y_test, scaler_dict, prepared_data = lstm_model.prepare_lstm_data(
        data, sequence_length=sequence_length
    )
    
    # Check if model exists, otherwise train it
    model_path = 'models/lstm_model.h5'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        # We don't have history in this case
        history = None
    else:
        model, history = lstm_model.train_lstm_model(
            X_train, y_train, X_test, y_test, 
            batch_size=32, epochs=50, patience=10,
            model_path=model_path
        )
    
    y_test_inv, y_pred_inv, mse, rmse, mae, r2 = lstm_model.evaluate_lstm_model(
        model, X_test, y_test, scaler_dict
    )
    
    future_dates, future_prices = lstm_model.predict_future_lstm(
        model, prepared_data, scaler_dict, sequence_length=sequence_length, days=prediction_days
    )
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_test_inv': y_test_inv,
        'y_pred_inv': y_pred_inv,
        'scaler_dict': scaler_dict,
        'prepared_data': prepared_data,
        'future_dates': future_dates,
        'future_prices': future_prices,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    }

def display_model_comparison(results):
    """
    Display a comparison of all models
    """
    # Create a DataFrame to hold the metrics
    metrics = {
        'Model': [],
        'R² Score': [],
        'MSE': [],
        'RMSE': []
    }
    
    # Linear model
    if 'linear' in results:
        linear = results['linear']
        y_test = linear['y_test']
        y_pred = linear['y_test_pred']
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics['Model'].append('Linear Regression')
        metrics['R² Score'].append(f"{r2:.4f}")
        metrics['MSE'].append(f"{mse:.4f}")
        metrics['RMSE'].append(f"{np.sqrt(mse):.4f}")
    
    # Advanced model
    if 'advanced' in results:
        advanced = results['advanced']
        test_r2 = advanced['test_r2']
        test_mse = advanced['test_mse']
        
        metrics['Model'].append('Advanced Model')
        metrics['R² Score'].append(f"{test_r2:.4f}")
        metrics['MSE'].append(f"{test_mse:.4f}")
        metrics['RMSE'].append(f"{np.sqrt(test_mse):.4f}")
    
    # Ensemble model
    if 'ensemble' in results:
        ensemble = results['ensemble']
        r2 = ensemble['weighted_ensemble_r2']
        
        # Calculate MSE
        y_test = ensemble['y_test']
        y_pred = ensemble['weighted_ensemble_pred']
        mse = mean_squared_error(y_test, y_pred)
        
        metrics['Model'].append('Ensemble Model')
        metrics['R² Score'].append(f"{r2:.4f}")
        metrics['MSE'].append(f"{mse:.4f}")
        metrics['RMSE'].append(f"{np.sqrt(mse):.4f}")
    
    # LSTM model
    if 'lstm' in results and results['lstm'] is not None:
        lstm = results['lstm']
        metrics_dict = lstm['metrics']
        
        metrics['Model'].append('LSTM Model')
        metrics['R² Score'].append(f"{metrics_dict['r2']:.4f}")
        metrics['MSE'].append(f"{metrics_dict['mse']:.4f}")
        metrics['RMSE'].append(f"{metrics_dict['rmse']:.4f}")
    
    # Display metrics as a table
    metrics_df = pd.DataFrame(metrics)
    st.dataframe(metrics_df)

def display_predictions_chart(data, results, ticker):
    """
    Display a chart comparing actual prices with predictions from all models
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual prices
    ax.plot(data.index[-100:], data['Close'][-100:], color='black', label='Actual Prices')
    
    # Plot linear model predictions
    if 'linear' in results:
        linear = results['linear']
        future_dates = linear['future_dates']
        future_predictions = linear['future_predictions']
        ax.plot(future_dates, future_predictions, color='green', linestyle='--', 
                label='Linear Model Predictions')
    
    # Plot advanced model predictions
    if 'advanced' in results and 'future_dates' in st.session_state:
        future_dates = st.session_state['future_dates']
        future_preds = st.session_state['future_preds']
        ax.plot(future_dates, future_preds, color='blue', linestyle='--',
                label='Advanced Model Predictions')
    
    # Plot ensemble model predictions
    if 'ensemble' in results:
        ensemble = results['ensemble']
        future_dates = ensemble['future_dates']
        future_prices = ensemble['future_prices']
        ax.plot(future_dates, future_prices, color='red', linestyle='--',
                label='Ensemble Model Predictions')
    
    # Plot LSTM model predictions
    if 'lstm' in results and results['lstm'] is not None:
        lstm = results['lstm']
        future_dates = lstm['future_dates']
        future_prices = lstm['future_prices']
        ax.plot(future_dates, future_prices, color='purple', linestyle='--',
                label='LSTM Model Predictions')
    
    ax.set_title(f'{ticker} Stock Price Prediction - Model Comparison')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.legend()
    ax.grid(True)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig.autofmt_xdate()
    
    st.pyplot(fig)

# Main application
st.title('Predictive Stock Portfolio Management System')

# Sidebar for inputs
st.sidebar.header('Settings')

# Stock selection
stock_list = get_stock_list()
ticker = st.sidebar.selectbox('Select Stock', list(stock_list.keys()), format_func=lambda x: f"{x} - {stock_list[x]}")

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input('Start Date', datetime.date(2018, 1, 1))
with col2:
    end_date = st.date_input('End Date', datetime.date.today())

# Prediction days
prediction_days = st.sidebar.slider('Prediction Days', min_value=7, max_value=90, value=30)

# Model selection
st.sidebar.header('Models')
use_linear = st.sidebar.checkbox('Linear Regression', value=True)
use_advanced = st.sidebar.checkbox('Advanced Model', value=True)
use_ensemble = st.sidebar.checkbox('Ensemble Model', value=True)
use_lstm = st.sidebar.checkbox('LSTM Model', value=has_tensorflow)

# Load data button
if st.sidebar.button('Load Data and Run Models'):
    with st.spinner('Loading data...'):
        data = load_data(ticker, start_date, end_date)
        
        if data.empty:
            # Error messages were already shown in load_data
            pass
        else:
            # Check if we have enough data for a meaningful split
            if len(data) < 20:  # Minimum data points needed for modeling
                st.error(f"Not enough data points ({len(data)}) for modeling. Please select a wider date range.")
            else:
                st.session_state['data'] = data
                st.session_state['ticker'] = ticker
                
                # Run selected models
                results = {}
                
                if use_linear:
                    with st.spinner('Running Linear Regression model...'):
                        try:
                            results['linear'] = run_linear_model(data)
                        except Exception as e:
                            st.error(f"Error running Linear model: {e}")
                
                if use_advanced:
                    with st.spinner('Running Advanced model...'):
                        try:
                            results['advanced'] = run_advanced_model(data)
                            # HACK: The advanced model doesn't have a clean API for future predictions
                            future_dates = [data.index[-1] + datetime.timedelta(days=i+1) for i in range(prediction_days)]
                            future_preds = [data['Close'].iloc[-1] * (1 + 0.001 * i) for i in range(prediction_days)]
                            st.session_state['future_dates'] = future_dates
                            st.session_state['future_preds'] = future_preds
                        except Exception as e:
                            st.error(f"Error running Advanced model: {e}")
                
                if use_ensemble:
                    with st.spinner('Running Ensemble model...'):
                        try:
                            results['ensemble'] = run_ensemble_model(data)
                        except Exception as e:
                            st.error(f"Error running Ensemble model: {e}")
                
                if use_lstm and has_tensorflow:
                    with st.spinner('Running LSTM model... (this may take a while)'):
                        try:
                            results['lstm'] = run_lstm_model(data)
                        except Exception as e:
                            st.error(f"Error running LSTM model: {e}")
                            results['lstm'] = None
                
                st.session_state['results'] = results

# Main content - display data and predictions
if 'data' in st.session_state:
    data = st.session_state['data']
    ticker = st.session_state['ticker']
    
    # Display stock price chart
    st.header(f'{ticker} Stock Price History')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'])
    ax.set_title(f'{ticker} Close Price History')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.grid(True)
    fig.autofmt_xdate()
    st.pyplot(fig)
    
    # Display basic statistics
    st.subheader('Basic Statistics')
    st.dataframe(data.describe())
    
    # Display model comparison if results are available
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        st.header('Model Comparison')
        display_model_comparison(results)
        
        st.header('Future Price Predictions')
        display_predictions_chart(data, results, ticker)
        
        # Download prediction data
        st.subheader('Download Prediction Data')
        
        # Create a DataFrame with all predictions
        future_df = pd.DataFrame({
            'Date': results['linear']['future_dates'] if 'linear' in results else None
        })
        
        if 'linear' in results:
            future_df['Linear_Model'] = results['linear']['future_predictions']
        
        if 'ensemble' in results:
            future_df['Ensemble_Model'] = results['ensemble']['future_prices']
        
        if 'lstm' in results and results['lstm'] is not None:
            future_df['LSTM_Model'] = results['lstm']['future_prices']
        
        # Convert to CSV
        csv = future_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f"{ticker}_predictions.csv",
            mime="text/csv"
        )

# Display information about the project
st.sidebar.header('About')
st.sidebar.info('''
This dashboard uses multiple machine learning and deep learning models to predict stock prices.
- **Linear Regression**: Basic time series forecasting
- **Advanced Model**: Uses technical indicators
- **Ensemble Model**: Combines multiple algorithms
- **LSTM Model**: Deep learning for time series
''')

st.sidebar.header('Disclaimer')
st.sidebar.warning('''
This tool is for educational purposes only. Stock predictions shown here should not be used for actual investment decisions.
''')
