import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile, upper_quantile):
        """
        Initialize the Winsorizer with lower and upper quantile thresholds.
        
        Args:
            lower_quantile (float): The lower quantile to cut (e.g., 0.05).
            upper_quantile (float): The upper quantile to cut (e.g., 0.95).
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X):
        """
        Compute the lower and upper quantiles for each feature in the training data.
        
        Args:
            X (array-like): Training data, shape (n_samples, n_features).
            
        Returns:
            self: Returns the instance itself.
        """
        X = np.asarray(X)
        self.lower_quantile_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_quantile_ = np.quantile(X, self.upper_quantile, axis=0)
        
        return self

    def transform(self, X):
        """
        Clip the data to the lower and upper quantiles computed during fit.
        
        Args:
            X (array-like): Data to be transformed.
            
        Returns:
            X_transformed (array-like): The winsorized data.
        """
        check_is_fitted(self, ['lower_quantile_', 'upper_quantile_'])
        X = np.asarray(X)
        X_transformed = np.clip(X, self.lower_quantile_, self.upper_quantile_)
        
        return X_transformed
