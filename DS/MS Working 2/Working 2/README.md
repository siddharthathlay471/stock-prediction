# Stock Prediction using Linear Regression

## Overview
This project uses Linear Regression to predict stock prices. It analyzes historical stock data to forecast future prices using a simple yet effective machine learning approach.

## Features
- Data acquisition from Yahoo Finance API
- Data preprocessing and feature engineering
- Linear Regression model training
- Performance evaluation (R² score, MSE)
- Future price prediction
- Visualization of results

## Linear Regression
Linear Regression models the relationship between a dependent variable (stock price) and independent variables (in this case, time) using the formula: Y = mx + c

Where:
- Y: dependent variable (stock price)
- x: independent variable (time/date)
- m: slope coefficient
- c: y-intercept

## Project Workflow
1. **Data Acquisition**: Download historical stock data
2. **Data Preparation**: Process data for training
3. **Model Training**: Train the Linear Regression model on 75% of data
4. **Model Evaluation**: Test on remaining 25% of data and calculate accuracy
5. **Future Prediction**: Forecast future stock prices
6. **Visualization**: Plot actual prices, predictions, and future forecasts

## Installation
```bash
git clone https://github.com/yourusername/stock-prediction.git
cd stock-prediction
pip install -r requirements.txt
```

## Usage
```bash
python stock_prediction.py
```

You can modify the ticker symbol, date range, and other parameters in the `main()` function.

## Example Results
The model typically achieves accuracy levels above 95% on test data, demonstrating the effectiveness of Linear Regression for stock price prediction over certain time periods.

## Technologies Used
- Python 3
- pandas, NumPy: Data manipulation
- Matplotlib: Visualization
- scikit-learn: Machine learning implementation
- yfinance: Stock data acquisition
