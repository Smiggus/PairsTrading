import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint

class PairScreener:
    def __init__(self, cleaned_data):
        self.cleaned_data = cleaned_data
        self.sector_mapping = self.get_sector_mapping()

    def get_sector_mapping(self):
        sector_mapping = {}
        for symbol in self.cleaned_data.columns:
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                sector = info.get('sector', 'Unknown')
                sector_mapping[symbol] = sector
            except Exception as e:
                print(f"Failed to get sector data for {symbol}: {e}")
                sector_mapping[symbol] = 'Unknown'
        return sector_mapping

    def find_pairs(self, confidence_level=0.99):
        pairs = []
        for stock1 in self.cleaned_data.columns:
            sector = self.sector_mapping.get(stock1)
            if sector and sector != 'Unknown':
                sector_stocks = [stock2 for stock2, sec in self.sector_mapping.items() if sec == sector and stock2 != stock1]
                for stock2 in sector_stocks:
                    returns1 = self.cleaned_data[stock1]
                    returns2 = self.cleaned_data[stock2]
                    
                    # Align the dataframes to ensure they have the same index
                    aligned_returns1, aligned_returns2 = returns1.align(returns2, join='inner')
                    
                    # Drop any rows with missing values
                    aligned_returns1.dropna(inplace=True)
                    aligned_returns2.dropna(inplace=True)
                    
                    if len(aligned_returns1) > 0 and len(aligned_returns2) > 0:
                        p_value = coint(aligned_returns1, aligned_returns2)[1]
                        if p_value < (1 - confidence_level):
                            pairs.append((stock1, stock2))
        return pairs

# Example usage
if __name__ == "__main__":
    # Example cleaned data (replace with actual cleaned data from StockDataETL)
    data = {
        'AAPL': np.random.randn(100),
        'MSFT': np.random.randn(100),
        'GOOGL': np.random.randn(100),
        'AMZN': np.random.randn(100),
        'TSLA': np.random.randn(100)
    }
    cleaned_data = pd.DataFrame(data)

    # Initialize the Pair Screener with cleaned data
    pair_screener = PairScreener(cleaned_data)
    pairs = pair_screener.find_pairs(confidence_level=0.99)

    print("Pairs with cointegration at 99% confidence level:")
    for pair in pairs:
        print(pair)
