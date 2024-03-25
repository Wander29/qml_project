from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pennylane import numpy as np
from sklearn.decomposition import PCA
import single_model_train
import qcnn_estimator
import data
import pandas as pd
import os
from datetime import datetime

samples = data.data_load_and_split(["rock", "country"])

estimator = qcnn_estimator.QCNNEstimator()
pipeline = Pipeline(steps=[
  ("scaler", MinMaxScaler((0, np.pi / 2))),
  ("pca", PCA(n_components=8)),
  ("model", estimator)
])

#--- Grid Search for hyperparameters
grid_params = { 'model__stride_c':list(range(2,4)),
                'model__filter_p':["!*", "*!"]
              }

grid = GridSearchCV(pipeline, grid_params, cv=3, n_jobs=8, verbose=True, refit=True)

grid.fit(samples.x_train, samples.y_train.values)

# Get the best model
best_circuit = grid.best_estimator_
# Print the best hyperparameters
print('=========================================[Best Hyperparameters info]=====================================')
# summarize best
print('Best score: %.3f'  % grid.best_score_)
print('Best Config: %s' % grid.best_params_)
print('==========================================================================================================')

best_circuit.score(samples.x_test, samples.y_test.values)
results = pd.DataFrame(grid.cv_results_)

os.makedirs('results/', exist_ok=True)
date = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
results.to_csv(f"results/out_{date}.csv", index=False)

