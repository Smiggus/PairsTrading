import yfinance as yf
import pandas as pd
import cupy as cp
from statsmodels.tsa.stattools import coint
import numpy as np
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PairScreener:
    def __init__(self, cleaned_data):
        self.cleaned_data = cleaned_data
        self.sector_mapping = self.get_sector_mapping()

    def get_sector_mapping(self):
        sector_mapping = {}
        logging.info("Retrieving sector information for each stock.")
        for symbol in tqdm(self.cleaned_data.columns, desc="Retrieving sectors"):
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                sector = info.get('sector', 'Unknown')
                sector_mapping[symbol] = sector
            except Exception as e:
                logging.warning(f"Failed to get sector data for {symbol}: {e}")
                sector_mapping[symbol] = 'Unknown'
        return sector_mapping

    def find_pairs(self, confidence_level=0.99):
        pairs = []
        logging.info("Starting the search for cointegrated pairs.")
        total_combinations = sum(len([stock2 for stock2, sec in self.sector_mapping.items() if sec == sector and stock2 != stock1])
                                 for stock1 in self.cleaned_data.columns
                                 for sector in [self.sector_mapping.get(stock1)]
                                 if sector and sector != 'Unknown')

        with tqdm(total=total_combinations, desc="Finding pairs") as pbar:
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
                            # Convert to CuPy arrays for GPU computation
                            cp_returns1 = cp.asarray(aligned_returns1.values)
                            cp_returns2 = cp.asarray(aligned_returns2.values)

                            # Perform the cointegration test using CuPy
                            coint_stat, p_value, _ = coint(cp_returns1.get(), cp_returns2.get())
                            correlation = np.corrcoef(cp_returns1.get(), cp_returns2.get())[0, 1]

                            if p_value < (1 - confidence_level):
                                pairs.append((stock1, stock2, coint_stat, p_value, correlation))
                        pbar.update(1)

        logging.info(f"Found {len(pairs)} pairs with cointegration at {confidence_level * 100}% confidence level.")
        
        # Create DataFrame from pairs and sort by cointegration level
        pairs_df = pd.DataFrame(pairs, columns=['Stock 1', 'Stock 2', 'Cointegration Stat', 'P-Value', 'Correlation'])
        pairs_df.sort_values(by='Cointegration Stat', ascending=True, inplace=True)
        
        return pairs_df
'''
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
    pairs_df = pair_screener.find_pairs(confidence_level=0.99)

    print("Pairs with cointegration at 99% confidence level:")
    print(pairs_df)
'''