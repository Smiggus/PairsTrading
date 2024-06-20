import numpy as np
import pandas as pd
import yfinance as yf

class StockDataETL:
    ''' The Data Pipeline for downloading, cleaning and transforming all the data for pairs processing. 
        Currently pulls just adjusted close for all universe securities, across the date range. 
    '''
    def __init__(self, symbols, start_date='2018-01-01', end_date='2023-01-01', missing_threshold=0.1):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.missing_threshold = missing_threshold

    def download_data(self):
        data_frames = []
        for symbol in self.symbols:
            try:
                ticker_data = yf.download(symbol, start=self.start_date, end=self.end_date)['Adj Close']
                data_frames.append(ticker_data.rename(symbol))
            except Exception as e:
                print(f"Failed to download data for {symbol}: {e}. Skipping...")
        return pd.concat(data_frames, axis=1)

    def clean_data(self, data):
        # Make a copy of the DataFrame to avoid modifying the original data
        data = data.copy()

        # Drop tickers with too many missing values
        data = data.dropna(thresh=int(data.shape[0] * (1 - self.missing_threshold)), axis=1)

        # Handle remaining missing data: fill forward, then fill backward
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

        # Ensure no inf or nan values are present
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        return data

    def calculate_returns(self, data):
        returns = data.pct_change().dropna()
        return returns

    def run_etl_pipeline(self):
        raw_data = self.download_data()
        cleaned_data = self.clean_data(raw_data)

        # Check if the data is empty after cleaning
        if cleaned_data.empty:
            raise ValueError("Data is empty after cleaning. Please check the data source and handling steps.")

        returns = self.calculate_returns(cleaned_data)

        # Check if the returns DataFrame is empty
        if returns.empty:
            raise ValueError("Returns data is empty after calculating percentage changes. Please check the data source and handling steps.")

        return returns
