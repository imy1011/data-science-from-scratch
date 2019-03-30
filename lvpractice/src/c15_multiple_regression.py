'''
Created on Jul 26, 2017

@author: loanvo
'''
from c04_linear_algebra import scalar_multiply, augmented_matrix, dot_product, vector_add
from c08_gradient_descent import minimize_stochastic
import random
from c14_simple_linear_regression import total_sum_of_squares
from c06_probability import normal_cdf
from functools import partial


"""
MULTIPLE REGRESSION
"""
def predict(x_i, beta):
    """Assume that the first element of each x_i is 1"""
    return dot_product(x_i, beta)

def error(x_i_y_i, beta):
    x_i = x_i_y_i[:-1]
    y_i = x_i_y_i[-1]
    return y_i - predict(x_i, beta)

def squared_error(x_i_y_i, beta):
    return error(x_i_y_i, beta) ** 2

def squared_error_gradient(x_i_y_i, beta):
    """ The gradient (with respect to beta)"""
    x_i = x_i_y_i[:-1]
    return scalar_multiply(-2 * error(x_i_y_i, beta), x_i)

def estimate_beta(x, y = None):
    if y is None: # if no y is provided, by default, it is considered that y is included into x i.e. the last column of x is y
        data = x
    else:
        data = augmented_matrix(x,y)
    beta_0 = [random.random() for _ in x[0]]
    beta = minimize_stochastic(squared_error, 
                               squared_error_gradient, 
                               data,
                               beta_0,
                               0.001)
    return beta

def multiple_r_squared(x, y, beta):
    return 1 - sum([squared_error(x_i + [y_i], beta) 
                    for x_i, y_i in zip(x, y)]) / total_sum_of_squares(y)
                    

"""
BOOTSTRAP
"""
                  
def bootstrap_sample(data):
    """randomly samples len(data) elements WITH REPLACEMENT"""
    return [random.choice(data) for _ in range(len(data))]
    
def bootstrap_statistic(data, stats_fn, num_samples):
    """evaluates stats_fn on num_samples bootstrap samples from data"""
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]


"""
STANDARD ERROR OF COEFFICIENTS
"""
def estimate_sample_beta(sample):
    """sample is a list of pairs (x_i, y_i)"""
    return estimate_beta(sample)

def p_value(beta_hat_j, sigma_hat_j):
    if beta_hat_j>0:
        return 2 * (1 - normal_cdf(beta_hat_j / sigma_hat_j))
    else:
        return 2 * normal_cdf(beta_hat_j / sigma_hat_j)
    
"""
REGULARIZATION
"""
# alpha is a *hyperparameter* controlling how harsh the penalty is
# sometimes it's called "lambda" but that already means something in Python
def ridge_penalty(beta, alpha):
    return alpha * dot_product(beta[1:], beta[1:])

def squared_error_ridge(x_i_y_i, beta, alpha):
    """estimate error plus ridge penalty on beta"""
    return squared_error(x_i_y_i, beta) + ridge_penalty(beta, alpha)

def ridge_penalty_gradient(beta, alpha):
    """gradient of just the ridge penalty"""
    return [0] + scalar_multiply(2 * alpha, beta[1:])

def squared_error_ridge_gradient(x_i_y_i, beta, alpha):
    """the gradient corresponding to the ith squared error term
    including the ridge penalty"""
    return vector_add(squared_error_gradient(x_i_y_i, beta), ridge_penalty_gradient(beta, alpha))

def estimate_beta_ridge(x, y = None, alpha = 0.001):
    """use gradient descent to fit a ridge regression
    with penalty alpha"""
    theta_0 = [random.random() for _ in x[0]]
    if y is None:
        data = x
    else:
        data = augmented_matrix(x, y)
    return minimize_stochastic(partial(squared_error_ridge, alpha = alpha), 
                               partial(squared_error_ridge_gradient, alpha = alpha), 
                               data, 
                               theta_0, 
                               0.001)

def lasso_penalty(beta, alpha):
    return alpha * sum(abs(beta_i) for beta_i in beta[1:])