import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import math
import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding

def data_embedding(X, embedding_type='Angle'):
    if embedding_type == 'Amplitude':
        AmplitudeEmbedding(X, wires=range(8), normalize=True)
    elif embedding_type == 'Angle':
        AngleEmbedding(X, wires=range(8), rotation='Y')

def data_load_and_process(dataset, N, classes):
    df = pd.read_csv("/home/ludovicowan/Files/sync/uni-sync/Unifi_corsi/QML/progetto_petruccione_audio/quarto/data/GTZan_dataset/features_30_sec.csv") 
    df = df.drop(columns=['filename'])
    df = df.drop(columns=['length'])
    X = df.drop('label', axis=1).values
    y = df['label'].values

    # scale for PCA
    scaler = MinMaxScaler(feature_range=(0, 1))
    ## @TODO for image?
    # x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0  # normalize the data

    x_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=N)
    x_pca = pca.fit_transform(x_scaled)

    # scale for angle embedding
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    x_final = scaler.fit_transform(x_pca)

    # split data
    x_train, x_test, y_train, y_test = train_test_split(
            x_final, y, test_size=0.3, random_state=0)

    # binary classes
    x_train_filter = np.where((y_train == classes[0]) | (y_train == classes[1]))
    x_test_filter = np.where((y_test == classes[0]) | (y_test == classes[1]))

    Y_train, Y_test = y_train[x_train_filter], y_test[x_test_filter]
    X_train, X_test = x_train[x_train_filter], x_test[x_test_filter]

    Y_train_bin = [1 if y == classes[0] else 0 for y in Y_train]
    Y_test_bin  = [1 if y == classes[0] else 0 for y in Y_test]

    return X_train, X_test, Y_train, Y_test