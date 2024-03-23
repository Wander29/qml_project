from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import single_model_train
import qcnn_estimator

scaler_for_pca = MinMaxScaler(feature_range=(0, 1))
scaler_for_angle_emb = MinMaxScaler(feature_range=(0, np.pi))
# params = [N,1,1,"*!"]
pca = PCA(n_components=8)
cnn = QCNNEstimator()
pipeline = Pipeline(steps=[
  ("scaler_pca", scaler_for_pca), 
  ("pca", pca),
  ("scaler_angle", scaler_for_angle_emb), 
  ("model", cnn)
])

#--- Grid Search for hyperparameters
grid_params = { 'model__N':[8], 
                'model__stride_c':list(range(1,7)),
                'model__stride_p':list(range(0,3)),
                'model__filter_p':["even"]
              }
              
grid = GridSearchCV(pipeline, grid_params, cv=5, n_jobs=-1, verbose=True, refit=True)
grid.fit(X, y)


# Other hyper parameters to try:
# mask patterns: *!, !*, !*!, *!*, 01, 10
# strides, steps and offsets for both Qmask and Qcycle 
# share_weights
# boundary conditions