import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import os
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
from newspaper import Article
import yfinance as yf

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentAnalyzer:
    """
    A class to retrieve and analyze news sentiment for stocks
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the SentimentAnalyzer
        
        Parameters:
        - api_key: API key for news services (optional)
        """
        self.api_key = api_key
        self.sia = SentimentIntensityAnalyzer()
        
    def get_news_alphavantage(self, ticker, days=30):
        """
        Get news from AlphaVantage API
        
        Parameters:
        - ticker: Stock symbol
        - days: Number of days of news to retrieve
        
        Returns:
        - DataFrame with news data
        """
        if not self.api_key:
            print("Warning: No API key provided for AlphaVantage")
            return pd.DataFrame()
            
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.api_key}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if 'feed' not in data:
                print(f"Error: {data.get('Information', 'No data returned from AlphaVantage')}")
                return pd.DataFrame()
                
            news_items = []
            
            for item in data['feed']:
                news_items.append({
                    'date': item.get('time_published', ''),
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'source': item.get('source', ''),
                    'url': item.get('url', ''),
                    'sentiment_score': item.get('overall_sentiment_score', None)
                })
            
            df = pd.DataFrame(news_items)
            
            # Convert date format
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S')
            
            # Filter by date range
            start_date = datetime.now() - timedelta(days=days)
            df = df[df['date'] >= start_date]
            
            return df
            
        except Exception as e:
            print(f"Error getting news from AlphaVantage: {e}")
            return pd.DataFrame()
    
    def get_news_yahoo(self, ticker, days=30):
        """
        Get news from Yahoo Finance
        
        Parameters:
        - ticker: Stock symbol
        - days: Number of days of news to retrieve
        
        Returns:
        - DataFrame with news data
        """
        try:
            # Use yfinance to get news
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                return pd.DataFrame()
                
            news_items = []
            
            for item in news:
                # Convert Unix timestamp to datetime
                date = datetime.fromtimestamp(item['providerPublishTime'])
                
                news_items.append({
                    'date': date,
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'source': item.get('publisher', ''),
                    'url': item.get('link', '')
                })
            
            df = pd.DataFrame(news_items)
            
            # Filter by date range
            start_date = datetime.now() - timedelta(days=days)
            df = df[df['date'] >= start_date]
            
            return df
            
        except Exception as e:
            print(f"Error getting news from Yahoo Finance: {e}")
            return pd.DataFrame()
    
    def analyze_sentiment_vader(self, text):
        """
        Analyze sentiment using VADER
        
        Parameters:
        - text: Text to analyze
        
        Returns:
        - Dictionary with sentiment scores
        """
        if not text or pd.isna(text):
            return {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}
            
        return self.sia.polarity_scores(text)
    
    def analyze_sentiment_textblob(self, text):
        """
        Analyze sentiment using TextBlob
        
        Parameters:
        - text: Text to analyze
        
        Returns:
        - Dictionary with polarity and subjectivity
        """
        if not text or pd.isna(text):
            return {'polarity': 0, 'subjectivity': 0}
            
        blob = TextBlob(text)
        return {'polarity': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity}
    
    def get_article_content(self, url):
        """
        Get the full content of an article from URL
        
        Parameters:
        - url: URL of the article
        
        Returns:
        - Article text
        """
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            print(f"Error getting article content: {e}")
            return ""
    
    def get_news_with_sentiment(self, ticker, days=30, source='yahoo', use_full_text=False):
        """
        Get news and analyze sentiment
        
        Parameters:
        - ticker: Stock symbol
        - days: Number of days of news to retrieve
        - source: News source ('yahoo' or 'alphavantage')
        - use_full_text: Whether to analyze the full article text
        
        Returns:
        - DataFrame with news and sentiment scores
        """
        # Get news data
        if source == 'alphavantage' and self.api_key:
            news_df = self.get_news_alphavantage(ticker, days)
        else:
            news_df = self.get_news_yahoo(ticker, days)
        
        if news_df.empty:
            print(f"No news found for {ticker}")
            return pd.DataFrame()
        
        # Get full article text if requested
        if use_full_text:
            print("Getting full article text (this may take a while)...")
            news_df['full_text'] = news_df['url'].apply(self.get_article_content)
            
            # Use full text for sentiment analysis if available
            news_df['text_for_sentiment'] = news_df.apply(
                lambda row: row['full_text'] if row['full_text'] else row['summary'], axis=1
            )
        else:
            # Use title + summary for sentiment analysis
            news_df['text_for_sentiment'] = news_df['title'] + " " + news_df['summary']
        
        # Analyze sentiment using VADER
        print("Analyzing sentiment...")
        news_df['vader_sentiment'] = news_df['text_for_sentiment'].apply(self.analyze_sentiment_vader)
        news_df['compound_score'] = news_df['vader_sentiment'].apply(lambda x: x['compound'])
        news_df['neg_score'] = news_df['vader_sentiment'].apply(lambda x: x['neg'])
        news_df['neu_score'] = news_df['vader_sentiment'].apply(lambda x: x['neu'])
        news_df['pos_score'] = news_df['vader_sentiment'].apply(lambda x: x['pos'])
        
        # Analyze sentiment using TextBlob
        news_df['textblob_sentiment'] = news_df['text_for_sentiment'].apply(self.analyze_sentiment_textblob)
        news_df['polarity'] = news_df['textblob_sentiment'].apply(lambda x: x['polarity'])
        news_df['subjectivity'] = news_df['textblob_sentiment'].apply(lambda x: x['subjectivity'])
        
        return news_df
    
    def aggregate_daily_sentiment(self, news_df):
        """
        Aggregate sentiment scores by day
        
        Parameters:
        - news_df: DataFrame with news and sentiment scores
        
        Returns:
        - DataFrame with daily sentiment scores
        """
        if news_df.empty:
            return pd.DataFrame()
            
        # Convert date to date only (no time)
        news_df['date_only'] = news_df['date'].dt.date
        
        # Group by date and calculate average sentiment scores
        daily_sentiment = news_df.groupby('date_only').agg({
            'compound_score': 'mean',
            'neg_score': 'mean',
            'neu_score': 'mean',
            'pos_score': 'mean',
            'polarity': 'mean',
            'subjectivity': 'mean',
            'title': 'count'
        }).reset_index()
        
        # Rename columns
        daily_sentiment = daily_sentiment.rename(columns={'title': 'news_count'})
        
        # Convert date_only back to datetime
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date_only'])
        daily_sentiment = daily_sentiment.drop('date_only', axis=1)
        
        return daily_sentiment
    
    def merge_sentiment_with_stock_data(self, sentiment_df, stock_data):
        """
        Merge sentiment data with stock price data
        
        Parameters:
        - sentiment_df: DataFrame with sentiment data
        - stock_data: DataFrame with stock price data
        
        Returns:
        - Merged DataFrame
        """
        if sentiment_df.empty:
            return stock_data.copy()
            
        # Convert index to datetime if it's not already
        stock_data_reset = stock_data.reset_index()
        
        # Make sure the date column has the right name
        if 'date' not in stock_data_reset.columns and 'Date' in stock_data_reset.columns:
            stock_data_reset = stock_data_reset.rename(columns={'Date': 'date'})
        
        # Merge dataframes
        merged_df = pd.merge(stock_data_reset, sentiment_df, on='date', how='left')
        
        # Fill missing sentiment values
        sentiment_cols = ['compound_score', 'neg_score', 'neu_score', 'pos_score', 'polarity', 'subjectivity', 'news_count']
        merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(0)
        
        # Set date as index again
        merged_df = merged_df.set_index('date')
        
        return merged_df
    
    def visualize_sentiment_vs_price(self, merged_df, ticker="STOCK"):
        """
        Visualize sentiment scores vs. stock price
        
        Parameters:
        - merged_df: DataFrame with stock and sentiment data
        - ticker: Stock symbol
        """
        plt.figure(figsize=(14, 10))
        
        # Plot stock price
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(merged_df.index, merged_df['Close'], 'b-', label='Close Price')
        ax1.set_ylabel('Price')
        ax1.set_title(f'{ticker} Stock Price and News Sentiment')
        ax1.grid(True)
        
        # Plot sentiment compound score
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.bar(merged_df.index, merged_df['compound_score'], color='g', alpha=0.6, label='VADER Compound')
        ax2.bar(merged_df.index, merged_df['polarity'], color='r', alpha=0.3, label='TextBlob Polarity')
        ax2.set_ylabel('Sentiment Score')
        ax2.set_title('News Sentiment Scores')
        ax2.grid(True)
        ax2.legend()
        
        # Plot news count
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.bar(merged_df.index, merged_df['news_count'], color='purple', alpha=0.6)
        ax3.set_ylabel('News Count')
        ax3.set_xlabel('Date')
        ax3.set_title('Number of News Articles')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{ticker}_sentiment_analysis.png')
        
    def calculate_sentiment_features(self, stock_data, ticker, days=30, source='yahoo', use_full_text=False):
        """
        Calculate sentiment features for stock prediction models
        
        Parameters:
        - stock_data: DataFrame with stock data
        - ticker: Stock symbol
        - days: Number of days of news to retrieve
        - source: News source ('yahoo' or 'alphavantage')
        - use_full_text: Whether to analyze the full article text
        
        Returns:
        - DataFrame with stock data and sentiment features
        """
        # Get news with sentiment
        news_df = self.get_news_with_sentiment(ticker, days, source, use_full_text)
        
        if news_df.empty:
            print(f"No sentiment data available for {ticker}")
            # Add empty sentiment columns to stock_data
            stock_data_copy = stock_data.copy()
            sentiment_cols = ['compound_score', 'neg_score', 'pos_score', 'news_count', 
                             'compound_score_ma5', 'sentiment_change']
            for col in sentiment_cols:
                stock_data_copy[col] = 0
            return stock_data_copy
        
        # Aggregate by day
        daily_sentiment = self.aggregate_daily_sentiment(news_df)
        
        # Merge with stock data
        merged_df = self.merge_sentiment_with_stock_data(daily_sentiment, stock_data)
        
        # Calculate additional sentiment features
        # Moving average of sentiment
        merged_df['compound_score_ma5'] = merged_df['compound_score'].rolling(window=5).mean().fillna(0)
        
        # Sentiment change
        merged_df['sentiment_change'] = merged_df['compound_score'].diff().fillna(0)
        
        # Sentiment momentum (rate of change)
        merged_df['sentiment_momentum'] = merged_df['compound_score'].pct_change(3).fillna(0)
        
        # Extremes of sentiment
        merged_df['sentiment_extreme'] = merged_df['compound_score'].abs()
        
        # Visualize 
        self.visualize_sentiment_vs_price(merged_df, ticker)
        
        return merged_df

def main():
    # Example usage
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Get stock data
    print(f"Downloading {ticker} stock data from {start_date} to {end_date}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Initialize sentiment analyzer
    # Replace with your actual API key if you have one
    api_key = os.environ.get('ALPHAVANTAGE_API_KEY', None)
    analyzer = SentimentAnalyzer(api_key=api_key)
    
    # Get news with sentiment and merge with stock data
    print(f"Getting news sentiment for {ticker}...")
    data_with_sentiment = analyzer.calculate_sentiment_features(stock_data, ticker, days=30)
    
    # Display the data
    print("\nStock data with sentiment features:")
    print(data_with_sentiment.tail())
    
    print("\nCorrelation between sentiment and stock price change:")
    data_with_sentiment['price_change'] = data_with_sentiment['Close'].pct_change()
    correlation = data_with_sentiment[['price_change', 'compound_score', 'polarity']].corr()
    print(correlation)
    
    print("Done!")

if __name__ == "__main__":
    main()
