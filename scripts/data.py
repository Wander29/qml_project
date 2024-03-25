from collections import namedtuple
# import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd
from pennylane import numpy as np

PATH_DATA = "/home/ludovicowan/Files/datasets/GTZan_dataset/features_30_sec.csv"
Samples = namedtuple("samples", ["x_train", "x_test", "y_train", "y_test"])

def data_load_and_split(genres):
  data = pd.read_csv(PATH_DATA)
  # remove filename and length columns
  data = data.drop(columns=["filename", "length"])
  # filter data
  data = data[data["label"].isin(genres)]
  # set label to 0 or 1
  data["label"] = data["label"].map({genres[0]: 0, genres[1]: 1})
  # specify target and features
  target = "label"
  X, y = data.drop(columns=[target]), data[target]
  # create train test split
  samples_raw = Samples(*train_test_split(X, y, test_size=0.3, random_state=42))
  return samples_raw

def data_load_and_process(genres):
    samples_raw = data_load_and_split(genres)
    
    # setup preprocessing pipeline
    pipeline = Pipeline(
        [
            (
                "scaler",
                MinMaxScaler((0, np.pi / 2)),
            ),
            ("pca", PCA(8)),
        ]
    )
    samples_preprocessed = Samples(
        pipeline.fit_transform(samples_raw.x_train),
        pipeline.transform(samples_raw.x_test),
        samples_raw.y_train,
        samples_raw.y_test,
    )
    return samples_preprocessed