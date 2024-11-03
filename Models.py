###################### CLUSTERING ##########################

######### k-meams ###########

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, f1_score, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn_extra.cluster import KMedoids
from utils import treat_final_df
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, Birch, SpectralClustering, AgglomerativeClustering



methods = ['GMM', 'KMeans', 'Spectral', 'Agglomerative', 'Birch','PAM']
index = ['Gaussian Mixture Models', 'KMeans', 'Spectral', 'Agglomerative', 'Birch','PAM']
metrics_clustering = pd.DataFrame(columns=['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score', 'f1_score'], index=index)
directory = 'images'



def apply_clustering(df, method, num_clusters=2, percentil_seguridad=95,initial_weights=None,verbose=True):
    """
    Apply clustering method to the dataset, predict clusters for unlabeled samples, and visualize the results.
    
    Parameters:
    df: DataFrame containing the dataset including the 'Malignancy' column.
    method: The clustering method to use ('GMM', 'KMeans', 'Birch', 'Spectral', 'Agglomerative').
    num_clusters: The number of clusters.
    initial_weights: Initial weights for GMM, default is None.

    
    Returns:
    Updated DataFrame, cluster scores (silhouette, calinski-harabasz, davies-bouldin), and f1-score.
    """
    # Separate features and labels
    X = df.drop(columns=['Malignancy'])
    y = df['Malignancy']
    df_before = df.copy()

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Choose clustering method
    if method == 'GMM':
        model = GaussianMixture(n_components=num_clusters, random_state=42, weights_init=initial_weights)
    elif method == 'KMeans':
        model = KMeans(n_clusters=num_clusters, random_state=42)
    elif method == 'Birch':
        model = Birch(n_clusters=num_clusters)
    elif method == 'Spectral':
        model = SpectralClustering(n_clusters=num_clusters, random_state=42, affinity='nearest_neighbors')
    elif method == 'Agglomerative':
        model = AgglomerativeClustering(n_clusters=num_clusters)
    elif method == 'PAM': 
        model =  KMedoids(n_clusters=num_clusters, random_state=42)
    else:
        raise ValueError("Invalid clustering method provided.")

    model.fit(X_train_scaled)
    si, ch, db = None, None, None
    
    # Predict clusters only for unlabeled examples (value 3)
    X_unlabeled = df[df['Malignancy'] == 3].drop(columns=['Malignancy'])
    if not X_unlabeled.empty:
        X_unlabeled_scaled = scaler.transform(X_unlabeled)
        unlabeled_clusters = model.fit_predict(X_unlabeled_scaled) if method in ['Spectral', 'Agglomerative'] else model.predict(X_unlabeled_scaled)

        # Calculate cluster scores
        si = silhouette_score(X_unlabeled_scaled, unlabeled_clusters)
        ch = calinski_harabasz_score(X_unlabeled_scaled, unlabeled_clusters)
        db = davies_bouldin_score(X_unlabeled_scaled, unlabeled_clusters)
        print(f'SI:{si:.4f}, ch:{ch:.4f}, db:{si:.4f}')
        # Assign new labels to the predicted clusters
        cluster_to_class = {0: 0, 1: 1}
        df.loc[df['Malignancy'] == 3, 'Malignancy'] = [cluster_to_class[cluster] for cluster in unlabeled_clusters]
        if  method in ['PAM', 'KMeans']:

            # Calcular distancias al centroide más cercano
            distances = model.transform(X_unlabeled_scaled)
            min_distances = distances.min(axis=1)
            
            # Crear DataFrame de distancias
            df_distances = pd.DataFrame({
                'Index': X_unlabeled.index,
                'Cluster': unlabeled_clusters,
                'Distance_to_Nearest_Centroid': min_distances
            })
        else: 
            df_distances = pd.DataFrame(columns=['Index', 'Cluster', 'Distance_to_Nearest_Centroid'])

    else:
        df_distances = pd.DataFrame(columns=['Index', 'Cluster', 'Distance_to_Nearest_Centroid'])

    # Calculate F1 score for the training data (excluding unlabeled examples)
    y_true = y_train[y_train != 3]
    X_known = X_train_scaled[y_train != 3]
    y_pred = model.fit_predict(X_known) if method in ['Spectral', 'Agglomerative'] else model.predict(X_known)

    
    if method in ['PAM', 'KMeans']:

        # Cálculo de distancias para puntos conocidos
        known_distances = model.transform(X_known)
        min_known_distances = known_distances.min(axis=1)

        # Filtrar puntos "seguros"
        treshold = np.percentile(min_known_distances, percentil_seguridad)
        indices_seguro = min_known_distances <= treshold
        y_true_seguro = y_true[indices_seguro]
        y_pred_seguro = y_pred[indices_seguro]

        # Calcular métricas sobre puntos "seguros"
        accuracy_seguro = accuracy_score(y_true_seguro, y_pred_seguro)
        f1_seguro = f1_score(y_true_seguro, y_pred_seguro, average='weighted')
        if verbose :
            print(f"Modelo {method} Accuracy seguro: {accuracy_seguro} ")
            print(f"Modelo {method} F1 seguro: {f1_seguro}")
    else: 

        f1_seguro = f1_score(y_true, y_pred, average='weighted')
        accuracy_seguro=accuracy_score(y_true, y_pred)
        if verbose :
            print(f'F1 en {method} :{f1_seguro:.4F}')
            print(f'Accuracy en {method} :{accuracy_seguro:.4F}')

    return df, si, ch, db, df_distances,f1_seguro,accuracy_seguro
