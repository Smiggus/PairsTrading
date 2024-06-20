import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import coint
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from joblib import Parallel, delayed

class CointegrationML:
    def __init__(self, returns):
        self.returns = returns

    def autoencode_and_cluster(self, n_clusters=10, significance_level=0.05, n_init=10):
        # Standardize returns
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(self.returns)

        # Define an autoencoder model
        input_dim = scaled_returns.shape[1]
        encoding_dim = 2

        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim, activation='relu')(input_layer)
        decoder = Dense(input_dim, activation='sigmoid')(encoder)

        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mse')

        # Train the autoencoder
        autoencoder.fit(scaled_returns, scaled_returns, epochs=50, batch_size=32, shuffle=True, validation_split=0.1)

        # Encode the returns data
        encoder_model = Model(inputs=input_layer, outputs=encoder)
        encoded_data = encoder_model.predict(scaled_returns)

        # Perform KMeans clustering on the encoded data
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)  
        kmeans.fit(encoded_data)
        labels = kmeans.labels_

        # Visualize the clusters using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(encoded_data)

        plt.figure(figsize=(10, 6))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('Clustering of S&P 500 Stocks Based on Encoded Returns')
        plt.show()

        return labels

    def find_cointegrated_pairs(self, labels, significance_level=0.05):
        unique_labels = np.unique(labels)
        pairs = []
        for label in unique_labels:
            cluster_stocks = self.returns.columns[labels == label]
            cluster_data = self.returns[cluster_stocks]
            if cluster_data.shape[1] < 2:
                continue  # Skip clusters with less than 2 stocks
            pairs.extend(self._find_cointegrated_pairs_in_cluster(cluster_data, significance_level))
        return pairs


    def _find_cointegrated_pairs_in_cluster(self, cluster_data, significance_level):
        keys = cluster_data.columns
        pairs = []

        # Use parallel processing to speed up the cointegration tests
        results = Parallel(n_jobs=-1)(delayed(self._coint_test)(cluster_data[keys[i]], cluster_data[keys[j]], significance_level) 
                                    for i in range(len(keys)) for j in range(i+1, len(keys)))

        # Check if the results length matches the expected number of pairs
        expected_pairs = len(keys) * (len(keys) - 1) // 2
        if len(results) != expected_pairs:
            raise ValueError(f"Expected {expected_pairs} cointegration test results but got {len(results)}.")

        # Extract pairs from the results
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                if results.pop(0):
                    pairs.append((keys[i], keys[j]))
        return pairs

    def _coint_test(self, stock1, stock2, significance_level):
        result = coint(stock1, stock2)
        p_value = result[1]
        return p_value < significance_level
