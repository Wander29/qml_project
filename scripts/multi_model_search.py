from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pennylane import numpy as np
from sklearn.decomposition import PCA
from single_model_train import (
  U_ansatz_conv_a, U_ansatz_conv_g,
  U_ansatz_pool_1, U_ansatz_pool_2
)
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


# Other hyper parameters to try:
# mask patterns: *!, !*, !*!, *!*, 01, 10
# strides, steps and offsets for both Qmask and Qcycle 
# share_weights
# boundary conditions
# ansatzes for convolution and pooling

#--- Grid Search for hyperparameters
grid_params = { 
                'model__stride_c':list(range(1,8)),
                # 'model__stride_c':[4],
                # 'model__step_c':list(range(1,8)),
                'model__step_c':[1],
                # 'model__offset_c':list(range(1,8)),
                'model__offset_c':[0],
                'model__share_weights':[True, False],
                
                'model__filter_p':["!*","*!", "!*!", "*!*", "01", "10"], #left, right, outside, inside, 01, 10#
                'model__stride_p':list(range(0,4)),
                # 'model__filter_p':["!*",],
                # 'model__ansatz_c':[U_ansatz_conv_a, U_ansatz_conv_g],
                'model__ansatz_c':[U_ansatz_conv_a],
                # 'model__ansatz_p':[U_ansatz_pool_1, U_ansatz_pool_2],
                'model__ansatz_p':[U_ansatz_pool_1],
              }

grid = GridSearchCV(pipeline, grid_params, cv=3, n_jobs=8, verbose=True, refit=True)

# start fit
import time
start_time = time.perf_counter()
grid.fit(samples.x_train, samples.y_train.values)
print("--- %.2f seconds ---" % (time.perf_counter() - start_time))

# Get the best model
best_circuit = grid.best_estimator_
accuracy_best_circuit = best_circuit.score(samples.x_test, samples.y_test.values)
# Print the best hyperparameters
print('=========================================[Best Hyperparameters info]=====================================')
# summarize best
print('Best training score: %.3f'  % grid.best_score_)
print('Accuracy best model: %.3f'  % accuracy_best_circuit)
print('Best Config: %s' % grid.best_params_)
print('Optimized symbols: %s' % best_circuit.named_steps.model.symbols)
print('==========================================================================================================')

# print grid search results
results = pd.DataFrame(grid.cv_results_)
os.makedirs('results/', exist_ok=True)
date = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
results.to_csv(f"results/out_{date}.csv", index=False)

