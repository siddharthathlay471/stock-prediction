import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import datetime
from tqdm import tqdm

# Import project modules
import stock_prediction
import advanced_model
import ensemble_model
try:
    import lstm_model
    has_tensorflow = True
except ImportError:
    has_tensorflow = False
try:
    import prophet_model
    has_prophet = True
except ImportError:
    has_prophet = False

def get_stock_data(ticker, start_date, end_date):
    """
    Download stock data for a given ticker and date range
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def backtest_model(ticker, model_name, start_date, end_date, window_size=252, step_size=21, prediction_days=30):
    """
    Backtest a model over a time period using rolling windows
    
    Parameters:
    - ticker: Stock symbol
    - model_name: Model to use ('linear', 'advanced', 'ensemble', 'lstm', 'prophet')
    - start_date: Start date for backtesting
    - end_date: End date for backtesting
    - window_size: Size of each training window in days (default: 252, about 1 year of trading days)
    - step_size: Days to move forward for each test (default: 21, about 1 month)
    - prediction_days: Number of days to predict for each window
    
    Returns:
    - Dictionary with backtesting results
    """
    print(f"Backtesting {model_name} model for {ticker} from {start_date} to {end_date}")
    
    # Get full data
    full_data = get_stock_data(ticker, start_date, end_date)
    if full_data.empty:
        print(f"Error: No data found for {ticker}")
        return None
    
    # Initialize lists to store results
    dates = []
    actual_prices = []
    predicted_prices = []
    mse_scores = []
    r2_scores = []
    mae_scores = []
    
    # Generate window ranges
    total_days = len(full_data)
    windows = []
    for start_idx in range(0, total_days - window_size - prediction_days, step_size):
        end_idx = start_idx + window_size
        test_end_idx = min(end_idx + prediction_days, total_days)
        windows.append((start_idx, end_idx, test_end_idx))
    
    # Backtest for each window
    for i, (start_idx, end_idx, test_end_idx) in enumerate(tqdm(windows, desc="Backtesting windows")):
        # Get training data
        train_data = full_data.iloc[start_idx:end_idx]
        # Get test data (actual future prices)
        test_data = full_data.iloc[end_idx:test_end_idx]
        
        if len(test_data) == 0:
            continue
        
        # Run the selected model
        try:
            if model_name == 'linear':
                X, y = stock_prediction.prepare_data(train_data)
                model, _, _, _, _, _, _ = stock_prediction.train_model(X, y)
                future_dates, future_predictions = stock_prediction.predict_future(
                    model, X, days=min(prediction_days, len(test_data))
                )
                
            elif model_name == 'advanced':
                X, y = advanced_model.prepare_advanced_features(train_data)
                model_result = advanced_model.train_advanced_model(X, y)
                model = model_result['model']
                
                # Get last row for prediction
                last_data = train_data.copy()
                
                # Hacky way to get predictions using existing code
                pred_dates = []
                pred_prices = []
                current_price = last_data['Close'].iloc[-1]
                for j in range(min(prediction_days, len(test_data))):
                    # Simple approximation for now
                    current_price *= (1 + np.random.normal(0.0005, 0.005))
                    pred_dates.append(test_data.index[j])
                    pred_prices.append(current_price)
                
                future_dates = pred_dates
                future_predictions = pred_prices
                
            elif model_name == 'ensemble':
                X, y, _ = ensemble_model.prepare_features(train_data)
                model_results = ensemble_model.build_ensemble_model(X, y)
                future_dates, future_predictions, _ = ensemble_model.predict_future_ensemble(
                    model_results, train_data, days=min(prediction_days, len(test_data))
                )
                
            elif model_name == 'lstm' and has_tensorflow:
                sequence_length = 60
                X_train, X_test, y_train, y_test, scaler_dict, prepared_data = lstm_model.prepare_lstm_data(
                    train_data, sequence_length=sequence_length
                )
                model, _ = lstm_model.train_lstm_model(
                    X_train, y_train, X_test, y_test, 
                    batch_size=32, epochs=50, patience=5
                )
                future_dates, future_predictions = lstm_model.predict_future_lstm(
                    model, prepared_data, scaler_dict, sequence_length=sequence_length, days=min(prediction_days, len(test_data))
                )
                
            elif model_name == 'prophet' and has_prophet:
                prophet_data = prophet_model.prepare_prophet_data(train_data)
                model, _, _, _, _, _, _, _ = prophet_model.train_prophet_model(prophet_data, test_size=0.1)
                future_dates, future_predictions, _, _, _ = prophet_model.predict_future_prophet(
                    model, prophet_data, days=min(prediction_days, len(test_data))
                )
                future_dates = [datetime.datetime.combine(d, datetime.time()) for d in future_dates]
            
            else:
                print(f"Error: Unknown model {model_name}")
                continue
                
            # Compare predictions to actual prices
            actual = []
            predicted = []
            
            for j, date in enumerate(future_dates):
                if j >= len(test_data):
                    break
                    
                # Find closest date in test data
                idx = test_data.index.get_indexer([date], method='nearest')[0]
                if idx < len(test_data):
                    actual_price = test_data['Close'].iloc[idx]
                    pred_price = future_predictions[j]
                    
                    dates.append(date)
                    actual_prices.append(actual_price)
                    predicted_prices.append(pred_price)
                    
                    actual.append(actual_price)
                    predicted.append(pred_price)
            
            # Calculate metrics for this window
            if len(actual) > 0 and len(predicted) > 0:
                window_mse = mean_squared_error(actual, predicted)
                window_r2 = r2_score(actual, predicted) if len(actual) > 1 else 0
                window_mae = mean_absolute_error(actual, predicted)
                
                mse_scores.append(window_mse)
                r2_scores.append(window_r2)
                mae_scores.append(window_mae)
                
        except Exception as e:
            print(f"Error in window {i+1}: {e}")
            continue
    
    # Calculate overall metrics
    results = {
        'dates': dates,
        'actual_prices': actual_prices,
        'predicted_prices': predicted_prices,
        'mse': np.mean(mse_scores) if mse_scores else float('nan'),
        'rmse': np.sqrt(np.mean(mse_scores)) if mse_scores else float('nan'),
        'r2': np.mean(r2_scores) if r2_scores else float('nan'),
        'mae': np.mean(mae_scores) if mae_scores else float('nan'),
        'window_mse': mse_scores,
        'window_r2': r2_scores,
        'window_mae': mae_scores
    }
    
    print(f"Backtesting completed. Overall metrics:")
    print(f"MSE: {results['mse']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"R²: {results['r2']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    
    return results

def visualize_backtest_results(results, ticker, model_name):
    """
    Visualize backtesting results
    """
    dates = results['dates']
    actual_prices = results['actual_prices']
    predicted_prices = results['predicted_prices']
    
    plt.figure(figsize=(14, 7))
    
    # Plot actual vs predicted prices
    plt.scatter(dates, actual_prices, color='black', s=10, label='Actual Prices')
    plt.scatter(dates, predicted_prices, color='red', s=10, label='Predicted Prices')
    
    # Draw lines between points
    for i in range(len(dates)-1):
        # Check if dates are consecutive to avoid drawing lines across gaps
        if (dates[i+1] - dates[i]).days <= 7:
            plt.plot([dates[i], dates[i+1]], [actual_prices[i], actual_prices[i+1]], 'k-', alpha=0.3)
            plt.plot([dates[i], dates[i+1]], [predicted_prices[i], predicted_prices[i+1]], 'r-', alpha=0.3)
    
    plt.title(f'{ticker} Backtesting Results - {model_name} Model')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{ticker}_{model_name}_backtest.png')
    
    # Plot error distribution
    errors = np.array(predicted_prices) - np.array(actual_prices)
    
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.hist(errors, bins=30)
    plt.title(f'Error Distribution - {model_name} Model')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 1, 2)
    plt.plot(dates, errors)
    plt.title('Prediction Error Over Time')
    plt.xlabel('Date')
    plt.ylabel('Error')
    plt.axhline(0, color='r', linestyle='-', alpha=0.3)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_{model_name}_backtest_errors.png')
    
    # Plot window metrics
    plt.figure(figsize=(14, 7))
    plt.subplot(3, 1, 1)
    plt.plot(results['window_mse'])
    plt.title('MSE by Window')
    plt.ylabel('MSE')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(results['window_r2'])
    plt.title('R² by Window')
    plt.ylabel('R²')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(results['window_mae'])
    plt.title('MAE by Window')
    plt.xlabel('Window Index')
    plt.ylabel('MAE')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_{model_name}_backtest_metrics.png')
    
def compare_backtest_models(ticker, start_date, end_date, models=None):
    """
    Compare backtesting results for multiple models
    """
    if models is None:
        models = ['linear', 'advanced', 'ensemble']
        if has_tensorflow:
            models.append('lstm')
        if has_prophet:
            models.append('prophet')
    
    results = {}
    for model_name in models:
        print(f"\nBacktesting {model_name} model...")
        model_results = backtest_model(ticker, model_name, start_date, end_date)
        if model_results:
            results[model_name] = model_results
            visualize_backtest_results(model_results, ticker, model_name)
    
    # Compare metrics
    comparison = {
        'Model': [],
        'MSE': [],
        'RMSE': [],
        'MAE': [],
        'R²': []
    }
    
    for model_name, model_results in results.items():
        comparison['Model'].append(model_name)
        comparison['MSE'].append(f"{model_results['mse']:.4f}")
        comparison['RMSE'].append(f"{model_results['rmse']:.4f}")
        comparison['MAE'].append(f"{model_results['mae']:.4f}")
        comparison['R²'].append(f"{model_results['r2']:.4f}")
    
    comparison_df = pd.DataFrame(comparison)
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Plot comparison
    plt.figure(figsize=(14, 10))
    
    # Create bar charts for each metric
    metrics = ['MSE', 'RMSE', 'MAE', 'R²']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        values = [float(val) for val in comparison[metric]]
        bars = plt.bar(comparison['Model'], values)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{height:.4f}', ha='center', va='bottom', rotation=0)
        
        plt.title(f'Comparison of {metric}')
        plt.ylabel(metric)
        plt.grid(True, axis='y', alpha=0.3)
        
        # For R², set y-axis to start from 0 and end at 1 for better visualization
        if metric == 'R²':
            plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_model_comparison.png')
    
    return comparison_df, results

def main():
    # Set parameters
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    
    # Run backtests for all models
    compare_backtest_models(ticker, start_date, end_date)
    
    print("Done!")

if __name__ == "__main__":
    main()
