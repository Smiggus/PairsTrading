# Pairs Trading Strategy Backtesting #

This application is relatively simple. It consists of 3 components.

The Stock_DataPipe which is an ETL pipeline within python that uses yfinance. It downloads the adjusted close for a list of stocks, within the date range BOY 2017 to EOY 2023. Future updates will introduce modularity to this.

The pairbacktest file. This is the main backtesting logic. It takes 2 stocks that are found to be a pair, and backtests them historically to find their cumulative return. It plots this on a chart as well with results for easy tracing and look back.

The final is the pairscreen-test which is the main research notebook. Its also currently the way I interface with the algorithm. 
It contains methods for:
    - Downloading the data, cleaning the data and uploading the data to the TARS postgresql database. 
    - Reading the data from the database
    - Running gpu-accelerated correlation tests to find pairs within a defined universe. In this instance, the S&P500 constituents
    - Backtesting a pairs trading strategy using two defined tickers.
    - Backtesting the top 2000 pairs (modifiable in the processed_pairs[:2000] line).
    - Calculating Sharpe and Volatility for those pairs
    - Calculating Portfolio level Sharpe and Volatility for the entire strategy. 

**Last updated 20JUN24**