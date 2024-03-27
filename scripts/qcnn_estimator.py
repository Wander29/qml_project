from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from single_model_train import (U_ansatz_conv_a, U_ansatz_pool_1, qcnn_motif, train, net, accuracy)

class QCNNEstimator(BaseEstimator) :
  def __init__(self, ansatz_c=U_ansatz_conv_a, stride_c=1, step_c=1, offset_c=0, share_weights=True, ansatz_p=U_ansatz_pool_1, filter_p='!*'):
    # convolution params
    self.ansatz_c = ansatz_c
    self.stride_c = stride_c
    self.step_c = step_c
    self.offset_c = offset_c
    self.share_weights = share_weights
    # pooling params
    self.ansatz_p = ansatz_p
    self.filter_p = filter_p
    # motif
    self.hierq = qcnn_motif(ansatz_c=ansatz_c, conv_stride=stride_c, conv_step=step_c, conv_offset=offset_c, share_weights=share_weights, ansatz_p=ansatz_p, pool_filter=filter_p)
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
    self.symbols, loss = train(X, y, self.hierq)
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
    
    return net(self.hierq, self.symbols, X)

  def score(self, data, targets):
    return accuracy(self.predict(data), targets)

  def __sklearn_clone__(self):
    return self