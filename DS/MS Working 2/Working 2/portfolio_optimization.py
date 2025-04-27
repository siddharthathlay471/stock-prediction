import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from scipy.optimize import minimize
import seaborn as sns
from sklearn.covariance import ledoit_wolf

class PortfolioOptimizer:
    """
    A class for portfolio optimization using various strategies
    """
    
    def __init__(self, tickers=None, start_date=None, end_date=None):
        """
        Initialize the portfolio optimizer
        
        Parameters:
        - tickers: List of stock tickers
        - start_date: Start date for historical data
        - end_date: End date for historical data
        """
        self.tickers = tickers or []
        self.start_date = start_date or (datetime.datetime.now() - datetime.timedelta(days=365*3)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        
    def load_data(self, tickers=None, start_date=None, end_date=None):
        """
        Load historical price data for the specified tickers
        """
        if tickers:
            self.tickers = tickers
        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date
            
        print(f"Loading data for {len(self.tickers)} stocks from {self.start_date} to {self.end_date}...")
        
        # Download data
        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
        
        # Handle single ticker case
        if len(self.tickers) == 1:
            self.data = pd.DataFrame(self.data, columns=self.tickers)
            
        # Calculate returns
        self.returns = self.data.pct_change().dropna()
        
        # Calculate mean returns and covariance matrix
        self.mean_returns = self.returns.mean()
        
        # Use Ledoit-Wolf shrinkage for better covariance estimation
        lw_cov, _ = ledoit_wolf(self.returns)
        self.cov_matrix = pd.DataFrame(lw_cov, index=self.returns.columns, columns=self.returns.columns)
        
        print(f"Loaded data with {len(self.data)} days for each ticker")
        return self.data
        
    def plot_correlation_matrix(self):
        """
        Plot the correlation matrix of returns
        """
        if self.returns is None:
            print("No data loaded. Please call load_data() first.")
            return
            
        plt.figure(figsize=(12, 10))
        corr = self.returns.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                    vmin=-1, vmax=1, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        
    def calculate_portfolio_performance(self, weights):
        """
        Calculate portfolio performance metrics
        
        Parameters:
        - weights: Array of weights for each asset
        
        Returns:
        - Dictionary of portfolio performance metrics
        """
        # Convert weights to array
        weights = np.array(weights)
        
        # Calculate portfolio return
        port_return = np.sum(self.mean_returns * weights) * 252  # Annualized
        
        # Calculate portfolio volatility
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0% for simplicity)
        sharpe_ratio = port_return / port_volatility
        
        return {
            'return': port_return,
            'volatility': port_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def negative_sharpe(self, weights):
        """
        Negative Sharpe ratio for minimization
        """
        return -self.calculate_portfolio_performance(weights)['sharpe_ratio']
    
    def optimize_portfolio(self, risk_free_rate=0.0, target_return=None, target_volatility=None, method='sharpe'):
        """
        Optimize portfolio weights
        
        Parameters:
        - risk_free_rate: Risk-free rate for Sharpe ratio calculation
        - target_return: Target portfolio return (for minimum volatility at target return)
        - target_volatility: Target portfolio volatility (for maximum return at target volatility)
        - method: Optimization method ('sharpe', 'min_vol', 'target_return', 'target_vol')
        
        Returns:
        - Dictionary with optimization results
        """
        if self.returns is None:
            print("No data loaded. Please call load_data() first.")
            return None
            
        num_assets = len(self.tickers)
        args = ()
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/num_assets] * num_assets)
        
        # Weight constraints (sum to 1)
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Optimization based on selected method
        if method == 'sharpe':
            # Maximize Sharpe ratio
            result = minimize(self.negative_sharpe, initial_weights, method='SLSQP', 
                            bounds=bounds, constraints=constraint)
                            
        elif method == 'min_vol':
            # Minimize volatility
            def portfolio_volatility(weights):
                weights = np.array(weights)
                return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
                
            result = minimize(portfolio_volatility, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraint)
                            
        elif method == 'target_return' and target_return is not None:
            # Minimize volatility at target return
            def portfolio_volatility(weights):
                weights = np.array(weights)
                return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
                
            # Additional constraint for target return
            target_return_constraint = {
                'type': 'eq',
                'fun': lambda x: np.sum(self.mean_returns * x) * 252 - target_return
            }
            
            constraints = [constraint, target_return_constraint]
            
            result = minimize(portfolio_volatility, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
                            
        elif method == 'target_vol' and target_volatility is not None:
            # Maximize return at target volatility
            def negative_portfolio_return(weights):
                weights = np.array(weights)
                return -np.sum(self.mean_returns * weights) * 252
                
            # Additional constraint for target volatility
            def portfolio_volatility(weights):
                weights = np.array(weights)
                return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
                
            target_volatility_constraint = {
                'type': 'eq',
                'fun': lambda x: portfolio_volatility(x) - target_volatility
            }
            
            constraints = [constraint, target_volatility_constraint]
            
            result = minimize(negative_portfolio_return, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
        else:
            print(f"Invalid optimization method: {method}")
            return None
        
        # Extract optimal weights
        optimal_weights = result['x']
        
        # Calculate performance metrics
        performance = self.calculate_portfolio_performance(optimal_weights)
        
        # Create results dictionary
        optimization_results = {
            'weights': optimal_weights,
            'tickers': self.tickers,
            'return': performance['return'],
            'volatility': performance['volatility'],
            'sharpe_ratio': performance['sharpe_ratio'],
            'method': method,
            'success': result['success'],
            'message': result['message']
        }
        
        return optimization_results
    
    def generate_efficient_frontier(self, num_portfolios=1000):
        """
        Generate the efficient frontier
        
        Parameters:
        - num_portfolios: Number of random portfolios to generate
        
        Returns:
        - DataFrame with portfolio weights and performance metrics
        """
        if self.returns is None:
            print("No data loaded. Please call load_data() first.")
            return None
            
        num_assets = len(self.tickers)
        results = np.zeros((num_portfolios, num_assets + 3)) # +3 for return, volatility, sharpe
        weights_record = []
        
        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(num_assets)
            weights = weights / np.sum(weights)
            weights_record.append(weights)
            
            # Calculate portfolio performance
            performance = self.calculate_portfolio_performance(weights)
            
            results[i, 0] = performance['return']
            results[i, 1] = performance['volatility']
            results[i, 2] = performance['sharpe_ratio']
            
            for j in range(num_assets):
                results[i, j+3] = weights[j]
        
        # Convert to DataFrame
        columns = ['Return', 'Volatility', 'Sharpe']
        columns.extend(self.tickers)
        
        df = pd.DataFrame(results, columns=columns)
        
        # Get the optimal portfolios
        max_sharpe_idx = df['Sharpe'].idxmax()
        max_sharpe_port = df.iloc[max_sharpe_idx]
        
        min_vol_idx = df['Volatility'].idxmin()
        min_vol_port = df.iloc[min_vol_idx]
        
        # Generate the efficient frontier with optimization
        target_returns = np.linspace(min_vol_port['Return'], max_sharpe_port['Return'] * 1.2, 30)
        efficient_frontier = []
        
        for target in target_returns:
            res = self.optimize_portfolio(method='target_return', target_return=target)
            if res and res['success']:
                efficient_frontier.append({
                    'Return': res['return'],
                    'Volatility': res['volatility'],
                    'Sharpe': res['sharpe_ratio'],
                    'Weights': res['weights']
                })
        
        ef_df = pd.DataFrame(efficient_frontier)
        
        return {
            'random_portfolios': df,
            'max_sharpe': max_sharpe_port,
            'min_vol': min_vol_port,
            'efficient_frontier': ef_df
        }
    
    def plot_efficient_frontier(self, frontier_results):
        """
        Plot the efficient frontier
        
        Parameters:
        - frontier_results: Results from generate_efficient_frontier
        """
        df = frontier_results['random_portfolios']
        max_sharpe_port = frontier_results['max_sharpe']
        min_vol_port = frontier_results['min_vol']
        ef_df = frontier_results['efficient_frontier']
        
        plt.figure(figsize=(12, 8))
        
        # Plot random portfolios
        plt.scatter(df['Volatility'], df['Return'], c=df['Sharpe'], cmap='viridis', alpha=0.5)
        
        # Plot efficient frontier
        plt.plot(ef_df['Volatility'], ef_df['Return'], 'r--', linewidth=3, label='Efficient Frontier')
        
        # Plot max Sharpe portfolio
        plt.scatter(max_sharpe_port['Volatility'], max_sharpe_port['Return'], 
                   marker='*', color='r', s=300, label='Maximum Sharpe')
                   
        # Plot min volatility portfolio
        plt.scatter(min_vol_port['Volatility'], min_vol_port['Return'], 
                   marker='*', color='g', s=300, label='Minimum Volatility')
                   
        plt.title('Efficient Frontier')
        plt.xlabel('Volatility (Standard Deviation)')
        plt.ylabel('Expected Return')
        plt.colorbar(label='Sharpe Ratio')
        plt.legend()
        plt.savefig('efficient_frontier.png')
    
    def backtest_portfolio(self, weights, start_date=None, end_date=None):
        """
        Backtest a portfolio with specified weights
        
        Parameters:
        - weights: Array of weights for each asset
        - start_date: Start date for backtesting (within loaded data range)
        - end_date: End date for backtesting (within loaded data range)
        
        Returns:
        - DataFrame with portfolio performance
        """
        if self.data is None:
            print("No data loaded. Please call load_data() first.")
            return None
        
        # Convert weights to dictionary
        if isinstance(weights, np.ndarray):
            weights_dict = {ticker: weight for ticker, weight in zip(self.tickers, weights)}
        elif isinstance(weights, dict):
            weights_dict = weights
        else:
            print("Invalid weights format")
            return None
        
        # Filter data for backtest period
        if start_date:
            backtest_data = self.data[self.data.index >= start_date]
        else:
            backtest_data = self.data.copy()
            
        if end_date:
            backtest_data = backtest_data[backtest_data.index <= end_date]
        
        # Calculate portfolio value
        portfolio = pd.DataFrame(index=backtest_data.index)
        
        # Add each stock's contribution based on weight
        for ticker, weight in weights_dict.items():
            if ticker in backtest_data.columns:
                # Normalize price to start with weight * 100
                normalized_price = backtest_data[ticker] / backtest_data[ticker].iloc[0] * weight * 100
                portfolio[ticker] = normalized_price
            else:
                print(f"Warning: {ticker} not in data")
        
        # Total portfolio value
        portfolio['Total'] = portfolio.sum(axis=1)
        
        # Calculate daily returns
        portfolio['Daily_Return'] = portfolio['Total'].pct_change()
        
        # Calculate cumulative returns
        portfolio['Cumulative_Return'] = (1 + portfolio['Daily_Return']).cumprod() - 1
        
        # Calculate metrics
        total_return = portfolio['Total'].iloc[-1] / portfolio['Total'].iloc[0] - 1
        annual_return = ((1 + total_return) ** (252 / len(portfolio))) - 1
        daily_returns = portfolio['Daily_Return'].dropna()
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized
        sharpe = annual_return / volatility  # Assuming risk-free rate of 0%
        sortino = annual_return / (daily_returns[daily_returns < 0].std() * np.sqrt(252))
        max_drawdown = (portfolio['Total'] / portfolio['Total'].cummax() - 1).min()
        
        # Print metrics
        print(f"Backtest Period: {backtest_data.index[0]} to {backtest_data.index[-1]}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annual Return: {annual_return:.2%}")
        print(f"Annual Volatility: {volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Sortino Ratio: {sortino:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        
        # Create metrics dictionary
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown
        }
        
        return portfolio, metrics
    
    def plot_backtest_results(self, portfolio_df, ticker_benchmark=None):
        """
        Plot backtest results
        
        Parameters:
        - portfolio_df: Portfolio DataFrame from backtest_portfolio
        - ticker_benchmark: Optional ticker symbol to use as benchmark
        """
        plt.figure(figsize=(14, 10))
        
        # Plot total portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(portfolio_df['Total'], label='Portfolio')
        
        # Add benchmark if specified
        if ticker_benchmark and ticker_benchmark in self.data.columns:
            # Normalize benchmark to match portfolio starting value
            normalized_benchmark = self.data[ticker_benchmark] / self.data[ticker_benchmark].loc[portfolio_df.index[0]] * 100
            benchmark_slice = normalized_benchmark.loc[portfolio_df.index]
            plt.plot(benchmark_slice, label=f'{ticker_benchmark} (Benchmark)')
        
        plt.title('Portfolio Performance')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        drawdown = (portfolio_df['Total'] / portfolio_df['Total'].cummax() - 1) * 100
        plt.fill_between(portfolio_df.index, drawdown, 0, color='red', alpha=0.3)
        plt.title('Portfolio Drawdown')
        plt.ylabel('Drawdown (%)')
        plt.xlabel('Date')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('portfolio_backtest.png')
    
    def optimize_and_backtest(self, method='sharpe', backtest_split=0.3):
        """
        Optimize portfolio on training data and backtest on test data
        
        Parameters:
        - method: Optimization method ('sharpe', 'min_vol')
        - backtest_split: Portion of data to use for backtesting (0.3 = 30% of data)
        
        Returns:
        - Dictionary with optimization and backtest results
        """
        if self.data is None:
            print("No data loaded. Please call load_data() first.")
            return None
        
        # Split data into training and testing
        split_idx = int(len(self.data) * (1 - backtest_split))
        training_end_date = self.data.index[split_idx]
        
        print(f"Training data: {self.data.index[0]} to {training_end_date}")
        print(f"Testing data: {training_end_date} to {self.data.index[-1]}")
        
        # Store original data
        original_data = self.data.copy()
        original_returns = self.returns.copy()
        original_mean_returns = self.mean_returns.copy()
        original_cov_matrix = self.cov_matrix.copy()
        
        # Set data to training period
        self.data = original_data.loc[:training_end_date]
        self.returns = self.data.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        
        # Use Ledoit-Wolf shrinkage for better covariance estimation
        lw_cov, _ = ledoit_wolf(self.returns)
        self.cov_matrix = pd.DataFrame(lw_cov, index=self.returns.columns, columns=self.returns.columns)
        
        # Optimize on training data
        print(f"Optimizing portfolio using {method} method...")
        opt_result = self.optimize_portfolio(method=method)
        
        if not opt_result or not opt_result['success']:
            print("Optimization failed")
            return None
        
        # Restore original data
        self.data = original_data
        self.returns = original_returns
        self.mean_returns = original_mean_returns
        self.cov_matrix = original_cov_matrix
        
        # Backtest on test data
        print("Backtesting optimal portfolio...")
        portfolio_df, metrics = self.backtest_portfolio(
            opt_result['weights'],
            start_date=training_end_date
        )
        
        # Plot backtest results
        self.plot_backtest_results(portfolio_df, ticker_benchmark=self.tickers[0])
        
        # Create results dictionary
        results = {
            'optimization': opt_result,
            'backtest': {
                'portfolio': portfolio_df,
                'metrics': metrics
            }
        }
        
        # Print portfolio weights
        print("\nOptimal Portfolio Weights:")
        for ticker, weight in zip(self.tickers, opt_result['weights']):
            print(f"{ticker}: {weight:.4f} ({weight*100:.2f}%)")
        
        return results

def main():
    # Example usage
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'V', 'JPM', 'WMT', 'PG']
    start_date = "2018-01-01"
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Initialize portfolio optimizer
    optimizer = PortfolioOptimizer(tickers, start_date, end_date)
    
    # Load data
    optimizer.load_data()
    
    # Plot correlation matrix
    optimizer.plot_correlation_matrix()
    
    # Generate efficient frontier
    print("Generating efficient frontier...")
    frontier_results = optimizer.generate_efficient_frontier(num_portfolios=1000)
    
    # Plot efficient frontier
    optimizer.plot_efficient_frontier(frontier_results)
    
    # Print optimal portfolios
    print("\nMaximum Sharpe Ratio Portfolio:")
    max_sharpe = frontier_results['max_sharpe']
    print(f"Expected Return: {max_sharpe['Return']:.4f}")
    print(f"Volatility: {max_sharpe['Volatility']:.4f}")
    print(f"Sharpe Ratio: {max_sharpe['Sharpe']:.4f}")
    
    print("\nMinimum Volatility Portfolio:")
    min_vol = frontier_results['min_vol']
    print(f"Expected Return: {min_vol['Return']:.4f}")
    print(f"Volatility: {min_vol['Volatility']:.4f}")
    print(f"Sharpe Ratio: {min_vol['Sharpe']:.4f}")
    
    # Out-of-sample testing
    print("\nRunning out-of-sample optimization and backtesting...")
    results = optimizer.optimize_and_backtest(method='sharpe', backtest_split=0.3)
    
    print("Done!")

if __name__ == "__main__":
    main()
