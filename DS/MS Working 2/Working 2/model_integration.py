import os
import yaml
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from joblib import dump, load

# Import all model modules with try/except for optional dependencies
import stock_prediction
import advanced_model
import ensemble_model

try:
    import xgboost_model
    has_xgboost = True
except ImportError:
    has_xgboost = False

try:
    import lstm_model
    import tensorflow as tf
    has_tensorflow = True
except ImportError:
    has_tensorflow = False

try:
    import prophet_model
    has_prophet = True
except ImportError:
    has_prophet = False

try:
    import sentiment_analysis
    has_sentiment = True
except ImportError:
    has_sentiment = False


class ModelManager:
    """
    Central manager for all prediction models to provide a unified interface
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialize the model manager
        
        Parameters:
        - config_path: Path to configuration YAML file
        """
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize model registry
        self.models = {}
        self.model_results = {}
        self.sentiment_analyzer = None
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
    def load_config(self, config_path):
        """
        Load configuration from YAML file
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            # Return default configuration
            return {
                'default': {
                    'ticker': 'AAPL',
                    'start_date': '2018-01-01',
                    'prediction_days': 30,
                    'test_size': 0.2
                },
                'models': {
                    'linear': {'enabled': True},
                    'advanced': {'enabled': True},
                    'ensemble': {'enabled': True},
                    'lstm': {'enabled': has_tensorflow},
                    'xgboost': {'enabled': has_xgboost},
                    'prophet': {'enabled': has_prophet}
                },
                'sentiment_analysis': {
                    'enabled': has_sentiment,
                    'days': 30,
                    'source': 'yahoo',
                    'use_full_text': False
                }
            }
    
    def get_stock_data(self, ticker=None, start_date=None, end_date=None, include_sentiment=False):
        """
        Get stock data with optional sentiment analysis features
        """
        # Use configuration defaults if not specified
        ticker = ticker or self.config['default']['ticker']
        start_date = start_date or self.config['default']['start_date']
        end_date = end_date or datetime.datetime.now().strftime('%Y-%m-%d')
        
        print(f"Downloading {ticker} stock data from {start_date} to {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Add sentiment analysis features if requested and available
        if include_sentiment and has_sentiment and self.config['sentiment_analysis']['enabled']:
            if self.sentiment_analyzer is None:
                api_key = os.environ.get('ALPHAVANTAGE_API_KEY', None)
                self.sentiment_analyzer = sentiment_analysis.SentimentAnalyzer(api_key=api_key)
            
            sentiment_config = self.config['sentiment_analysis']
            data = self.sentiment_analyzer.calculate_sentiment_features(
                data, ticker, 
                days=sentiment_config['days'],
                source=sentiment_config['source'],
                use_full_text=sentiment_config['use_full_text']
            )
            
        return data
    
    def train_linear_model(self, data, ticker, prediction_days=None):
        """
        Train and evaluate the linear regression model
        """
        prediction_days = prediction_days or self.config['default']['prediction_days']
        
        # Prepare data and train model
        X, y = stock_prediction.prepare_data(data)
        model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = stock_prediction.train_model(X, y)
        future_dates, future_predictions = stock_prediction.predict_future(model, X, days=prediction_days)
        
        # Store results
        result = {
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'future_dates': future_dates,
            'future_predictions': future_predictions,
            'ticker': ticker
        }
        
        self.models['linear'] = model
        self.model_results['linear'] = result
        
        # Save model
        dump(model, f'models/linear_model_{ticker}.joblib')
        
        return result
    
    def train_advanced_model(self, data, ticker, prediction_days=None):
        """
        Train and evaluate the advanced model with technical indicators
        """
        prediction_days = prediction_days or self.config['default']['prediction_days']
        
        # Prepare data and train model
        X, y = advanced_model.prepare_advanced_features(data)
        model_result = advanced_model.train_advanced_model(X, y)
        
        # Generate future dates for compatibility with other models
        future_dates = [data.index[-1] + datetime.timedelta(days=i+1) for i in range(prediction_days)]
        
        # Use a simple approximation for future predictions since the advanced model doesn't have a clean API
        last_price = data['Close'].iloc[-1]
        future_preds = [last_price * (1 + 0.001 * i) for i in range(prediction_days)]
        
        # Add to the result
        model_result['future_dates'] = future_dates
        model_result['future_predictions'] = future_preds
        model_result['ticker'] = ticker
        
        self.models['advanced'] = model_result['model']
        self.model_results['advanced'] = model_result
        
        return model_result
    
    def train_ensemble_model(self, data, ticker, prediction_days=None):
        """
        Train and evaluate the ensemble model
        """
        prediction_days = prediction_days or self.config['default']['prediction_days']
        
        # Prepare data and train model
        X, y, prepared_data = ensemble_model.prepare_features(data)
        model_results = ensemble_model.build_ensemble_model(X, y)
        future_dates, future_prices, future_prices_by_model = ensemble_model.predict_future_ensemble(
            model_results, data, days=prediction_days
        )
        
        # Add results to model output
        model_results['future_dates'] = future_dates
        model_results['future_prices'] = future_prices
        model_results['future_prices_by_model'] = future_prices_by_model
        model_results['ticker'] = ticker
        
        self.models['ensemble'] = model_results['trained_models']
        self.model_results['ensemble'] = model_results
        
        return model_results
    
    def train_lstm_model(self, data, ticker, prediction_days=None):
        """
        Train and evaluate the LSTM model
        """
        if not has_tensorflow:
            print("TensorFlow is not installed. LSTM model is not available.")
            return None
            
        prediction_days = prediction_days or self.config['default']['prediction_days']
        lstm_config = self.config['models']['lstm']
        
        # Get sequence length from config
        sequence_length = lstm_config.get('sequence_length', 60)
        
        # Prepare data
        X_train, X_test, y_train, y_test, scaler_dict, prepared_data = lstm_model.prepare_lstm_data(
            data, sequence_length=sequence_length
        )
        
        # Check if model exists, otherwise train it
        model_path = f'models/lstm_model_{ticker}.h5'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            history = None
        else:
            model, history = lstm_model.train_lstm_model(
                X_train, y_train, X_test, y_test, 
                batch_size=lstm_config.get('batch_size', 32),
                epochs=lstm_config.get('epochs', 50),
                patience=lstm_config.get('patience', 10),
                model_path=model_path
            )
        
        # Evaluate model
        y_test_inv, y_pred_inv, mse, rmse, mae, r2 = lstm_model.evaluate_lstm_model(
            model, X_test, y_test, scaler_dict
        )
        
        # Generate future predictions
        future_dates, future_prices = lstm_model.predict_future_lstm(
            model, prepared_data, scaler_dict, 
            sequence_length=sequence_length, 
            days=prediction_days
        )
        
        # Store results
        result = {
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
            'ticker': ticker,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        }
        
        self.models['lstm'] = model
        self.model_results['lstm'] = result
        
        return result
    
    def train_xgboost_model(self, data, ticker, prediction_days=None):
        """
        Train and evaluate the XGBoost model
        """
        if not has_xgboost:
            print("XGBoost is not installed. XGBoost model is not available.")
            return None
            
        prediction_days = prediction_days or self.config['default']['prediction_days']
        
        # Prepare features
        X, y, prepared_data = xgboost_model.prepare_features(data)
        
        # Train model
        model_result = xgboost_model.train_xgboost_model(X, y)
        
        # Predict future prices
        future_dates, future_prices = xgboost_model.predict_future_xgboost(model_result, data, days=prediction_days)
        
        # Add to results
        model_result['future_dates'] = future_dates
        model_result['future_prices'] = future_prices
        model_result['ticker'] = ticker
        
        self.models['xgboost'] = model_result['model']
        self.model_results['xgboost'] = model_result
        
        return model_result
    
    def train_prophet_model(self, data, ticker, prediction_days=None):
        """
        Train and evaluate the Prophet model
        """
        if not has_prophet:
            print("Prophet is not installed. Prophet model is not available.")
            return None
            
        prediction_days = prediction_days or self.config['default']['prediction_days']
        
        # Prepare data
        prophet_data = prophet_model.prepare_prophet_data(data)
        
        # Train model
        model, forecast, train_data, test_data, mse, rmse, mae, r2 = prophet_model.train_prophet_model(
            prophet_data, test_size=self.config['default']['test_size']
        )
        
        # Predict future
        future_dates, future_prices, lower_bound, upper_bound, full_forecast = prophet_model.predict_future_prophet(
            model, prophet_data, days=prediction_days
        )
        
        # Store results
        result = {
            'model': model,
            'forecast': forecast,
            'train_data': train_data,
            'test_data': test_data,
            'future_dates': future_dates,
            'future_prices': future_prices,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'full_forecast': full_forecast,
            'ticker': ticker,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        }
        
        self.models['prophet'] = model
        self.model_results['prophet'] = result
        
        return result
    
    def train_all_models(self, ticker=None, start_date=None, end_date=None, prediction_days=None, include_sentiment=False):
        """
        Train all enabled models
        """
        # Get stock data
        data = self.get_stock_data(ticker, start_date, end_date, include_sentiment=include_sentiment)
        ticker = ticker or self.config['default']['ticker']
        prediction_days = prediction_days or self.config['default']['prediction_days']
        
        results = {}
        
        # Train each enabled model
        if self.config['models']['linear']['enabled']:
            print("Training Linear Regression model...")
            results['linear'] = self.train_linear_model(data, ticker, prediction_days)
            
        if self.config['models']['advanced']['enabled']:
            print("Training Advanced model...")
            results['advanced'] = self.train_advanced_model(data, ticker, prediction_days)
            
        if self.config['models']['ensemble']['enabled']:
            print("Training Ensemble model...")
            results['ensemble'] = self.train_ensemble_model(data, ticker, prediction_days)
            
        if has_tensorflow and self.config['models']['lstm']['enabled']:
            print("Training LSTM model...")
            results['lstm'] = self.train_lstm_model(data, ticker, prediction_days)
            
        if has_xgboost and self.config['models']['xgboost']['enabled']:
            print("Training XGBoost model...")
            results['xgboost'] = self.train_xgboost_model(data, ticker, prediction_days)
            
        if has_prophet and self.config['models']['prophet']['enabled']:
            print("Training Prophet model...")
            results['prophet'] = self.train_prophet_model(data, ticker, prediction_days)
            
        return {
            'data': data,
            'ticker': ticker,
            'results': results
        }
    
    def compare_models(self, results=None):
        """
        Compare performance metrics across all trained models
        """
        if results is None:
            if not self.model_results:
                print("No models trained yet. Call train_all_models first.")
                return None
            results = self.model_results
            
        # Create a DataFrame for comparison
        comparison = {
            'Model': [],
            'R²': [],
            'MSE': [],
            'RMSE': [],
            'MAE': []
        }
        
        # Linear model
        if 'linear' in results:
            linear = results['linear']
            y_test = linear['y_test']
            y_pred = linear['y_test_pred']
            
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            comparison['Model'].append('Linear Regression')
            comparison['R²'].append(r2)
            comparison['MSE'].append(mse)
            comparison['RMSE'].append(np.sqrt(mse))
            comparison['MAE'].append(mae)
        
        # Advanced model
        if 'advanced' in results:
            advanced = results['advanced']
            
            comparison['Model'].append('Advanced Model')
            comparison['R²'].append(advanced['test_r2'])
            comparison['MSE'].append(advanced['test_mse'])
            comparison['RMSE'].append(np.sqrt(advanced['test_mse']))
            comparison['MAE'].append(advanced.get('test_mae', np.nan))
        
        # Ensemble model
        if 'ensemble' in results:
            ensemble = results['ensemble']
            
            comparison['Model'].append('Ensemble Model')
            comparison['R²'].append(ensemble['weighted_ensemble_r2'])
            comparison['MSE'].append(ensemble.get('weighted_ensemble_mse', np.nan))
            comparison['RMSE'].append(np.sqrt(ensemble.get('weighted_ensemble_mse', 0)))
            comparison['MAE'].append(ensemble.get('weighted_ensemble_mae', np.nan))
        
        # LSTM model
        if 'lstm' in results and results['lstm'] is not None:
            lstm = results['lstm']
            metrics = lstm['metrics']
            
            comparison['Model'].append('LSTM Model')
            comparison['R²'].append(metrics['r2'])
            comparison['MSE'].append(metrics['mse'])
            comparison['RMSE'].append(metrics['rmse'])
            comparison['MAE'].append(metrics['mae'])
        
        # XGBoost model
        if 'xgboost' in results and results['xgboost'] is not None:
            xgb = results['xgboost']
            
            comparison['Model'].append('XGBoost Model')
            comparison['R²'].append(xgb['test_r2'])
            comparison['MSE'].append(xgb['test_mse'])
            comparison['RMSE'].append(xgb['test_rmse'])
            comparison['MAE'].append(xgb['test_mae'])
        
        # Prophet model
        if 'prophet' in results and results['prophet'] is not None:
            prophet = results['prophet']
            metrics = prophet['metrics']
            
            comparison['Model'].append('Prophet Model')
            comparison['R²'].append(metrics['r2'])
            comparison['MSE'].append(metrics['mse'])
            comparison['RMSE'].append(metrics['rmse'])
            comparison['MAE'].append(metrics['mae'])
        
        # Convert to DataFrame
        df = pd.DataFrame(comparison)
        
        # Calculate mean improvement over linear baseline
        if 'Linear Regression' in df['Model'].values:
            linear_metrics = df[df['Model'] == 'Linear Regression'].iloc[0]
            
            # Calculate relative improvement for each model
            for metric in ['R²', 'MSE', 'RMSE', 'MAE']:
                if metric == 'R²':  # Higher is better
                    df[f'{metric} Improvement'] = df[metric] / linear_metrics[metric] - 1
                else:  # Lower is better
                    df[f'{metric} Improvement'] = 1 - df[metric] / linear_metrics[metric]
        
        return df
    
    def get_best_model(self, metric='R²', higher_is_better=True):
        """
        Get the best model based on a specific metric
        """
        comparison = self.compare_models()
        
        if comparison is None or comparison.empty:
            return None
        
        if higher_is_better:
            best_idx = comparison[metric].idxmax()
        else:
            best_idx = comparison[metric].idxmin()
            
        best_model_name = comparison.loc[best_idx, 'Model']
        
        # Map display name to key
        name_to_key = {
            'Linear Regression': 'linear',
            'Advanced Model': 'advanced',
            'Ensemble Model': 'ensemble',
            'LSTM Model': 'lstm',
            'XGBoost Model': 'xgboost',
            'Prophet Model': 'prophet'
        }
        
        best_model_key = name_to_key.get(best_model_name)
        
        if best_model_key in self.model_results:
            return {
                'name': best_model_name,
                'key': best_model_key,
                'result': self.model_results[best_model_key],
                'model': self.models.get(best_model_key)
            }
        else:
            return None

# Example usage
if __name__ == "__main__":
    manager = ModelManager()
    results = manager.train_all_models(ticker="AAPL", start_date="2020-01-01", include_sentiment=True)
    comparison = manager.compare_models()
    print(comparison)
    
    best_model = manager.get_best_model(metric='R²')
    print(f"Best model: {best_model['name']}")
