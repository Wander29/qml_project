from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import single_model_train

class QCNNEstimator(BaseEstimator) :
  def __init__(self, stride_c=1, filter_p='!*'):
    self.stride_c = stride_c
    self.filter_p = filter_p
    self.hierq = single_model_train.qcnn_motif(stride_c, filter_p)
    self.symbols = []

  def fit(self, X, y, **kwargs):
    """Implementation of a fitting function.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,) or (n_samples, n_outputs)
        The target values (class labels in classification, real numbers in
        regression).

    Returns
    -------
    self : object
        Returns self.
    """
    # Check that X and y have correct shape
    X, y = check_X_y(X, y, accept_sparse=True)
    self.symbols, loss = single_model_train.train(X, y, self.hierq)
    self.is_fitted_ = True

    # `fit` should always return `self`
    return self

  def predict(self, X):
    """ Implementation of a predicting function.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The training input samples.

    Returns
    -------
    y : ndarray, shape (n_samples,)
        Returns the predicted results
    """
    X = check_array(X, accept_sparse=True) # Input validation
    check_is_fitted(self, 'is_fitted_') # Check if fit has been called
    
    return single_model_train.net(self.hierq, self.symbols, X)

  def score(self, data, targets):
    return single_model_train.accuracy(self.predict(data), targets)

  def __sklearn_clone__(self):
    return self