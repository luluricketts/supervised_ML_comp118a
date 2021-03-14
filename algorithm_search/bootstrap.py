# Bootstraps 5000 samples for training set and rest to test set

import numpy as np

def bootstrap(X, y, n_train=5000):
    """
    Creates training and testing sets with 5000 training examples
    sampled with replacement
    
    parameters
    ----------
    X: feature list
    y: targets
    n_train: number
    
    returns:
    --------
    X_train, X_test, y_train, y_test numpy arrays
    """
    
    n_samples = X.shape[0]
    train_inds = np.random.randint(n_samples, size=n_train)
    
    X_train = np.take(X, train_inds, axis=0)
    y_train = np.take(y, train_inds, axis=0)
    
    X_test = np.take(X, [i for i in range(n_samples) if i not in train_inds], axis=0)
    y_test = np.take(y, [i for i in range(n_samples) if i not in train_inds], axis=0)
    
    return X_train, X_test, y_train, y_test