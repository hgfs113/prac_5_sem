import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar


class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None,
                 feature_subsample_size=None, **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None
            then use recommendations.
        """
        self.n_estimators = n_estimators
        self.K = feature_subsample_size
        self.forest = []
        for j in range(n_estimators):
            self.forest.append(DecisionTreeRegressor(criterion='mse',
                               max_depth=max_depth, **trees_parameters))
        self.forest = np.array(self.forest)
        self.feat_subsamps = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """
        if self.K is None:
            self.K = X.shape[1] // 3
        a = np.arange(X.shape[1])
        for i in range(self.n_estimators):
            BootStrap = np.random.randint(0, X.shape[0],  X.shape[0])
            F = np.random.choice(a, size=self.K, replace=False)
            self.feat_subsamps.append(F)
            self.forest[i].fit(X[BootStrap[:, np.newaxis], F], y[BootStrap])

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        rez_y = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            rez_y += self.forest[i].predict(X[:, self.feat_subsamps[i]])
        return rez_y / self.n_estimators



class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        learning_rate : float
            Use learning_rate * gamma instead of gamma
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        pass
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        """
        pass

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pass
