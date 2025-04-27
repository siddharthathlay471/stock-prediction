import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import yfinance as yf
import datetime
from joblib import dump, load
import os

def get_stock_data(ticker, start_date, end_date):
    """
    Download stock data for a given ticker and date range
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def prepare_features(data, window_size=60):
    """
    Prepare features for XGBoost model with comprehensive technical indicators
    """
    df = data.copy()
    
    # Technical indicators
    
    # Moving averages
    for window in [5, 10, 20, 50, 200]:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
        df[f'MA_Volume{window}'] = df['Volume'].rolling(window=window).mean()
    
    # Exponential moving averages
    for window in [5, 12, 26, 50]:
        df[f'EMA{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    # Bollinger Bands
    for window in [10, 20, 50]:
        df[f'STD{window}'] = df['Close'].rolling(window=window).std()
        df[f'Upper_Band{window}'] = df[f'MA{window}'] + (df[f'STD{window}'] * 2)
        df[f'Lower_Band{window}'] = df[f'MA{window}'] - (df[f'STD{window}'] * 2)
        df[f'BB_Width{window}'] = (df[f'Upper_Band{window}'] - df[f'Lower_Band{window}']) / df[f'MA{window}']
        df[f'BB_Position{window}'] = (df['Close'] - df[f'Lower_Band{window}']) / (df[f'Upper_Band{window}'] - df[f'Lower_Band{window}'])
    
    # Relative Strength Index (RSI)
    for window in [7, 14, 21]:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df[f'RSI{window}'] = 100 - (100 / (1 + rs))
    
    # Price momentum
    for window in [1, 3, 5, 10, 21]:
        df[f'Price_Change_{window}d'] = df['Close'].pct_change(periods=window)
    
    # Volatility
    df['High_Low_Range'] = df['High'] - df['Low']
    df['Close_Open_Range'] = abs(df['Close'] - df['Open'])
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Historical volatility
    for window in [5, 10, 21, 63]:  # 1 week, 2 weeks, 1 month, 3 months
        df[f'Volatility_{window}d'] = df['Daily_Return'].rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    # Volume indicators
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Average True Range (ATR)
    for window in [7, 14, 21]:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'ATR{window}'] = tr.rolling(window=window).mean()
    
    # Commodity Channel Index (CCI)
    for window in [20]:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        ma_tp = typical_price.rolling(window=window).mean()
        mean_deviation = abs(typical_price - ma_tp).rolling(window=window).mean()
        df[f'CCI{window}'] = (typical_price - ma_tp) / (0.015 * mean_deviation)
    
    # Williams %R
    for window in [14, 21]:
        highest_high = df['High'].rolling(window=window).max()
        lowest_low = df['Low'].rolling(window=window).min()
        df[f'Williams_%R{window}'] = (highest_high - df['Close']) / (highest_high - lowest_low) * -100
    
    # Rate of Change (ROC)
    for window in [5, 10, 21]:
        df[f'ROC{window}'] = (df['Close'] / df['Close'].shift(window) - 1) * 100
    
    # Price distance from moving averages (as percentage)
    for ma in [5, 10, 20, 50, 200]:
        df[f'Distance_from_MA{ma}'] = (df['Close'] - df[f'MA{ma}']) / df[f'MA{ma}'] * 100
    
    # Moving Average Crossovers (binary indicators)
    df['MA5_cross_MA20'] = ((df['MA5'] > df['MA20']) & (df['MA5'].shift() <= df['MA20'].shift())).astype(int)
    df['MA20_cross_MA50'] = ((df['MA20'] > df['MA50']) & (df['MA20'].shift() <= df['MA50'].shift())).astype(int)
    df['MA50_cross_MA200'] = ((df['MA50'] > df['MA200']) & (df['MA50'].shift() <= df['MA200'].shift())).astype(int)
    
    # Add day of the week, month features
    df['day_of_week'] = pd.to_datetime(df.index).dayofweek
    df['month'] = pd.to_datetime(df.index).month
    df['quarter'] = pd.to_datetime(df.index).quarter
    
    # Target variable: next day's closing price
    df['Next_Close'] = df['Close'].shift(-1)
    
    # Add lag features
    for i in range(1, min(window_size + 1, 6)):  # Limit to 5 lags to avoid too many features
        df[f'Close_lag_{i}'] = df['Close'].shift(i)
        df[f'Volume_lag_{i}'] = df['Volume'].shift(i)
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    # Features and target
    columns_to_drop = ['Next_Close']
    if 'Adj Close' in df.columns:
        columns_to_drop.append('Adj Close')
    
    X = df.drop(columns_to_drop, axis=1)
    y = df['Next_Close']
    
    return X, y, df

def train_xgboost_model(X, y):
    """
    Train and tune XGBoost model for stock prediction
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'alpha': 1,
        'n_estimators': 1000,
        'seed': 42
    }
    
    # Number of boosting rounds
    num_rounds = 1000
    
    # Train model with early stopping
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(params, dtrain, num_rounds, evals=evals, 
                      early_stopping_rounds=50, verbose_eval=100)
    
    # Make predictions
    y_train_pred = model.predict(dtrain)
    y_test_pred = model.predict(dtest)
    
    # Evaluate model
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    
    print(f"XGBoost Model - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"Test MSE: {test_mse:.4f}, Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}")
    
    # Feature importance analysis
    importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save model
    model.save_model('models/xgboost_model.json')
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'importance': importance_df
    }

def predict_future_xgboost(model_result, data, days=30):
    """
    Predict future stock prices using the trained XGBoost model
    """
    model = model_result['model']
    
    # Prepare the last data point
    last_data = data.copy()
    X_last, _, _ = prepare_features(last_data)
    
    future_dates = [data.index[-1] + datetime.timedelta(days=i+1) for i in range(days)]
    future_prices = []
    
    # Start with the last row from the dataset
    last_row = X_last.iloc[-1:].copy()
    
    # Predict recursively for future days
    for i in range(days):
        # Create DMatrix for XGBoost
        dmatrix = xgb.DMatrix(last_row)
        # Predict next price
        next_price = model.predict(dmatrix)[0]
        
        # Store prediction
        future_prices.append(next_price)
        
        if i < days - 1:  # Don't update for the last day
            # Copy the row for the next day's prediction
            new_row = last_row.copy()
            
            # Update the 'Close' price (which will be used for the next prediction)
            new_row.loc[new_row.index[0], 'Close'] = next_price
            
            # Update lag features
            for j in range(5, 1, -1):  # Assuming we have 5 lag features
                if f'Close_lag_{j}' in new_row.columns and f'Close_lag_{j-1}' in new_row.columns:
                    new_row.loc[new_row.index[0], f'Close_lag_{j}'] = new_row.loc[new_row.index[0], f'Close_lag_{j-1}']
                    
            if 'Close_lag_1' in new_row.columns:
                new_row.loc[new_row.index[0], 'Close_lag_1'] = last_row.loc[last_row.index[0], 'Close']
            
            # Update Open, High, Low (simplistic approach)
            new_row.loc[new_row.index[0], 'Open'] = next_price
            new_row.loc[new_row.index[0], 'High'] = next_price * 1.01
            new_row.loc[new_row.index[0], 'Low'] = next_price * 0.99
            
            # For simplicity, keep other features unchanged
            # In a real implementation, you'd want to update all technical indicators
            
            # Use this row for the next prediction
            last_row = new_row
    
    return future_dates, future_prices

def plot_xgboost_results(data, model_result, future_dates=None, future_prices=None, ticker="STOCK"):
    """
    Visualize XGBoost model results and predictions
    """
    plt.figure(figsize=(14, 10))
    
    # Plot actual vs predicted for test set
    plt.subplot(2, 1, 1)
    
    X_test = model_result['X_test']
    y_test = model_result['y_test']
    y_test_pred = model_result['y_test_pred']
    
    test_dates = X_test.index
    
    plt.plot(test_dates, y_test, color='black', label='Actual Prices')
    plt.plot(test_dates, y_test_pred, color='blue', label='XGBoost Predictions')
    
    # Plot future predictions
    if future_dates is not None and future_prices is not None:
        plt.plot(future_dates, future_prices, 'r--', linewidth=2, label='Future Predictions')
    
    plt.title(f'{ticker} Stock Price - XGBoost Model Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    # Plot feature importance
    plt.subplot(2, 1, 2)
    
    importance_df = model_result['importance']
    top_features = importance_df.head(20)  # Top 20 features
    
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.title('Top 20 Feature Importance in XGBoost Model')
    plt.xlabel('Importance (Gain)')
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_xgboost_prediction.png')
    
    return plt

def main():
    # Set parameters
    ticker = "AAPL"
    start_date = "2015-01-01"
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Get data
    print(f"Downloading {ticker} stock data from {start_date} to {end_date}...")
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    # Prepare features
    print("Preparing features...")
    X, y, prepared_data = prepare_features(stock_data)
    
    # Train XGBoost model
    print("Training XGBoost model...")
    model_result = train_xgboost_model(X, y)
    
    # Predict future prices
    print("Predicting future prices...")
    future_dates, future_prices = predict_future_xgboost(model_result, stock_data, days=30)
    
    # Plot results
    print("Plotting results...")
    plot_xgboost_results(stock_data, model_result, future_dates, future_prices, ticker)
    
    print("Done!")

if __name__ == "__main__":
    main()
