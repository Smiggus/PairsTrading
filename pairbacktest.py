''' Introducing variables for the call'''
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

''' 
    Backtesting model for pairs trading strategy
    Takes 2 stocks, returns the cumulative return for the pair across start and end date.
    Adapting this code to automatically run on all the pairs in the universe.
    Start and end date must still be manually defined below. 
'''

def pairbt(stock1:str, stock2: str):
    # Step 1: Download historical data
    symbols = [stock1, stock2]
    data = yf.download(symbols, start='2012-01-01', end='2023-01-01')['Adj Close']

    # Step 2: Calculate the spread
    # Calculate log prices
    log_prices = np.log(data)
    # Calculate the spread using a linear combination of the log prices
    spread = log_prices[stock1] - log_prices[stock2]

    # Step 3: Calculate the Z-score of the spread
    def zscore(series):
        return (series - series.mean()) / np.std(series)

    spread_zscore = zscore(spread)

    # Step 4: Generate trading signals
    entry_threshold = 2
    exit_threshold = 1

    # Buy signal (long PEP, short KO)
    data['longs'] = spread_zscore < -entry_threshold
    # Sell signal (short PEP, long KO)
    data['shorts'] = spread_zscore > entry_threshold
    # Exit signal (close positions)
    data['exits'] = (spread_zscore > -exit_threshold) & (spread_zscore < exit_threshold)

    # Initialize positions
    data['positions'] = 0

    # Long positions
    data.loc[data['longs'], 'positions'] = 1
    # Short positions
    data.loc[data['shorts'], 'positions'] = -1
    # Exit positions
    data.loc[data['exits'], 'positions'] = 0

    # Forward fill positions to maintain them until an exit signal
    data['positions'] = data['positions'].replace(to_replace=0, method='ffill')

    # Step 5: Calculate strategy returns
    data['returns'] = data['positions'].shift(1) * (data[stock1].pct_change() - data[stock2].pct_change())

    # Calculate cumulative returns
    data['cumulative_returns'] = (1 + data['returns']).cumprod()

    # Convert cumulative returns to percentage
    data['cumulative_returns_pct'] = (data['cumulative_returns'] - 1) * 100

    # Plot cumulative returns
    plt.figure(figsize=(10, 5))
    plt.plot(data['cumulative_returns_pct'], label='Pairs Trading Strategy')
    plt.title('Pairs Trading Strategy Performance')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns (%)')
    plt.legend()
    plt.show()

    # Print final cumulative return in percentage
    print(f"Final cumulative return: {data['cumulative_returns_pct'].iloc[-1]:.2f}%")

    return data

def calculate_volatility(StratRet):
    """
    Calculate the annualized volatility from daily returns in StratRet DataFrame.
    
    Parameters:
    StratRet (DataFrame): DataFrame containing 'returns' column with daily returns.

    Returns:
    float: Annualized volatility of the returns.
    """
    # Calculate daily volatility
    daily_volatility = np.std(StratRet['returns'])

    # Annualize the daily volatility
    annual_volatility = daily_volatility * np.sqrt(252)

    return daily_volatility, annual_volatility

def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.04):
    """
    Calculate the annualized Sharpe ratio from daily returns.
    
    Parameters:
    daily_returns (Series or array): Series or array of daily returns.
    risk_free_rate (float): Annual risk-free rate (default is 0.01 or 1%).
    
    Returns:
    float: Annualized Sharpe ratio.
    """
    # Calculate the average daily return
    mean_daily_return = np.mean(daily_returns)
    
    # Annualize the mean daily return
    annualized_return = mean_daily_return * 252
    
    # Calculate the daily volatility
    daily_volatility = np.std(daily_returns)
    
    # Annualize the daily volatility
    annual_volatility = daily_volatility * np.sqrt(252)
    
    # Calculate the Sharpe ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / annual_volatility
    
    return sharpe_ratio