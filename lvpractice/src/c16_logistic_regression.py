'''
Created on Jul 26, 2017

@author: loanvo
'''

import math
from c04_linear_algebra import dot_product, vector_sum

# Logistic function
def logistic(x):
    return 1/(1 + math.exp(-x))
# Gradient of logistic function
def logistic_gradient(x):
    return logistic(x) * (1-logistic(x))
# Assume the likelihood of p(yi | xi, beta) = f(xi*beta) ** yi + [1 - f(xi*beta)] ** (1-yi)
# Logistic log likelihood function (input: a single data point) 
def logistic_log_likelihood_i(x_i_y_i, beta):
    y_i = x_i_y_i[-1]
    x_i = x_i_y_i[:-1]
    return y_i*math.log(logistic(dot_product(x_i,beta))) + \
        (1-y_i)*math.log(1- logistic(dot_product(x_i,beta)))
# Sum of Logistic log likelihood functions of all data points
def logistic_log_likelihood(x, y, beta):
    return sum([logistic_log_likelihood_i(x_i + [y_i], beta) for x_i, y_i in zip(x,y)])
# partial derivative of logistic log function of a single point input
def logistic_log_partial_ij(x_i, y_i, beta, j):
    """here i is the index of the data point, j the index of the derivative"""
    return (y_i - logistic(dot_product(x_i, beta)))*x_i[j]
# gradient of logistic log function (of a single point input)
def logistic_log_gradient_i(x_i_y_i, beta):
    """the gradient of the log likelihood corresponding to the ith data point"""
    y_i = x_i_y_i[-1]
    x_i = x_i_y_i[:-1]
    return [logistic_log_partial_ij(x_i, y_i, beta, j) for j, _ in enumerate(beta)]
# gradient of logistic log function
def logistic_log_gradient(x, y, beta):
    return vector_sum(*[logistic_log_gradient_i(x_i + [y_i], beta) for x_i, y_i in zip(x,y)])
