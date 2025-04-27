import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
import datetime

def get_stock_data(ticker, start_date, end_date):
    """
    Download stock data for a given ticker and date range
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def prepare_data(data):
    """
    Prepare data for linear regression model
    """
    # Reset index to make Date a column
    data = data.reset_index()
    
    # Convert Date to numeric for model training
    data['Date'] = mdates.date2num(data['Date'])
    
    # Create features and target
    X = data[['Date']]  # Features
    y = data['Close']   # Target (closing price)
    
    return X, y

def train_model(X, y):
    """
    Train a linear regression model and evaluate its performance
    """
    # Split data into training and testing sets (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate model
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    print(f"Train R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Model Accuracy: {test_r2 * 100:.2f}%")
    
    return model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred

def predict_future(model, data, days=30):
    """
    Predict stock prices for future dates
    """
    last_date = mdates.num2date(data['Date'].iloc[-1])
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days+1)]
    future_dates_num = mdates.date2num(future_dates)
    
    # Make predictions
    future_dates_df = pd.DataFrame({'Date': future_dates_num})
    future_predictions = model.predict(future_dates_df)
    
    return future_dates, future_predictions

def visualize_results(data, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, 
                     future_dates=None, future_predictions=None, ticker="STOCK"):
    """
    Visualize the results of the model
    """
    plt.figure(figsize=(12, 6))
    
    # Convert numeric dates back to datetime for plotting
    train_dates = mdates.num2date(X_train['Date'])
    test_dates = mdates.num2date(X_test['Date'])
    
    # Plot actual data - check if Date is already datetime
    if isinstance(data['Date'].iloc[0], (pd.Timestamp, datetime.datetime)):
        plt.plot(data['Date'], data['Close'], color='black', label='Actual Prices')
    else:
        plt.plot(mdates.num2date(data['Date']), data['Close'], color='black', label='Actual Prices')
    
    # Plot training predictions
    plt.scatter(train_dates, y_train_pred, color='green', label='Training Predictions', alpha=0.6)
    
    # Plot testing predictions
    plt.scatter(test_dates, y_test_pred, color='red', label='Testing Predictions', alpha=0.6)
    
    # Plot future predictions if available
    if future_dates is not None and future_predictions is not None:
        plt.plot(future_dates, future_predictions, 'b--', label='Future Predictions')
    
    plt.title(f'{ticker} Stock Price Prediction using Linear Regression')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis to show dates properly
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()  # Rotate date labels
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_prediction.png')
    plt.show()

def main():
    # Set parameters
    ticker = "AAPL"  # Apple stock
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    
    # Get data
    print(f"Downloading {ticker} stock data from {start_date} to {end_date}...")
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    # Prepare data
    X, y = prepare_data(stock_data)
    
    # Train model and evaluate
    print("Training model...")
    model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = train_model(X, y)
    
    # Predict future prices
    print("Predicting future prices...")
    future_dates, future_predictions = predict_future(model, X, days=30)
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(stock_data.reset_index(), X_train, X_test, y_train, y_test, 
                     y_train_pred, y_test_pred, future_dates, future_predictions, ticker)
    
    print("Done!")

if __name__ == "__main__":
    main()
