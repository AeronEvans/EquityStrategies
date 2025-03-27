import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import statistics

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_stock_data(api_key: str, symbol: str) -> dict:

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": "full"
    }
    
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    return response.json()

def preprocess_dataframe(data: dict) -> pd.DataFrame:


    time_series = data.get("Time Series (Daily)", {})
    
    df = pd.DataFrame.from_dict(time_series, orient='index')

    df.reset_index(inplace=True)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    df.sort_values('Date', inplace=True)
    
    return df

def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:

    # Simple Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    
    # Exponential Moving Averages
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_30'] = df['Close'].ewm(span=30, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    return df

def calculate_rsi(data: pd.Series, periods: int = 14) -> pd.Series:

    delta = data.diff()
    
    # Make two series: one for lower closes and one for higher closes
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    # Use exponential moving average
    ma_up = up.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
    
    relative_strength = ma_up / ma_down
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    
    return rsi

def calculate_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['Close'].rolling(window=20).std() * 2)
    
    return df

def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    return df

def calculate_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # True Range and Average True Range
    df['High_Low'] = df['High'] - df['Low']
    df['High_PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low_PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    df['True_Range'] = df[['High_Low', 'High_PrevClose', 'Low_PrevClose']].max(axis=1)
    df['ATR_14'] = df['True_Range'].rolling(window=14).mean()
    
    return df

def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:

    # Daily Returns
    df['Daily_Return'] = df['Close'].pct_change() * 100
    
    # Rate of Change
    df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Momentum
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    
    # Stochastic Oscillator
    lowest_low = df['Low'].rolling(window=14).min()
    highest_high = df['High'].rolling(window=14).max()
    df['Stochastic_K'] = ((df['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
    df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
    
    return df

def calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:

    # Volume Change
    df['Volume_Change'] = df['Volume'].pct_change() * 100
    
    # Price to Volume Ratio
    df['Price_Volume_Ratio'] = (df['Close'] * df['Volume']) / df['Volume'].rolling(window=20).mean()
    
    return df

def get_daily_performance(api_key: str, symbol: str) -> pd.DataFrame:

    try:
        # Fetch raw data
        raw_data = fetch_stock_data(api_key, symbol)
        
        # Preprocess data
        df = preprocess_dataframe(raw_data)
        
        # Calculate various indicators
        df = (df.pipe(calculate_moving_averages)
               .pipe(calculate_bollinger_bands)
               .pipe(calculate_macd)
               .pipe(calculate_volatility_indicators)
               .pipe(calculate_momentum_indicators)
               .pipe(calculate_volume_indicators))
        
        # Add RSI 
        df['RSI_14'] = calculate_rsi(df['Close'])
        return df
    
    except (KeyError, ValueError, requests.RequestException) as e:
        print(f"Error processing stock data: {e}")
        return None



def main():
    # Replace with your actual Alpha Vantage API key
    api_key = "UHQL6DBXKH7HQXQ3"
    symbol = "T"
    

    stock_data = get_daily_performance(api_key, symbol)#

    for row in stock_data.iterrows():
        print(row[1])
        break
        


if __name__ == "__main__":
    main()