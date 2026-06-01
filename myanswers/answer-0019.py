import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def segmentar_productos(df_productos, n_clusters):
    df = df_productos.copy()
    df['precio'] = np.abs(df['precio'])
    
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    
    return scaler.inverse_transform(kmeans.cluster_centers_)
