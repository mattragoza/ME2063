import numpy as np


def poly_basis(x, K):
    '''
    Polynomial basis functions.
    
    Args:
        x: (N, 1) input array.
        K: # of basis functions.
    Returns:
        (N, K) output array.
    '''
    return x ** np.arange(K)


def gauss_basis(x, K, s=0.2):
    '''
    Gaussian basis functions.
    
    Args:
        x: (N, 1) input array.
        K: # of basis functions.
    Returns:
        (N, K) output array.
    '''
    phi = np.ones((len(x), K))
    mu = np.linspace(0, 1, K - 1)
    #mu = np.linspace(0, 1, K + 1)[1:-1]
    #s = 1 / K
    phi[:,1:] = np.exp(-(x - mu)**2 / s**2)
    return phi


def linear_model(x, theta, basis):
    '''
    Evaluate linear regression model.
    
    Args:
        x: (N, 1) input array.
        theta: (K, 1) parameter array.
        basis: Basis function(s).
    Returns:
        (N, 1) output array.
    '''
    phi = basis(x, K=len(theta))
    return phi @ theta


def linear_fit(x, y, K, basis):
    '''
    Fit linear regression model.
    
    Args:
        x: (N, 1) input array.
        y: (N, 1) target array.
        K: # of basis functions.
        basis: Basis function(s).
    Returns:
        theta: (K, 1) parameter array.
    '''
    phi = basis(x, K)
    A, b = phi.T @ phi, phi.T @ y
    return np.linalg.solve(A, b)


def mse(y_pred, y_true):
    '''
    Mean squared error.
    
    Args:
        y_pred: (N, 1) output array.
        y_true: (N, 1) target array.
    Returns:
        Error value.
    '''
    return np.mean((y_pred - y_true)**2) / 2
