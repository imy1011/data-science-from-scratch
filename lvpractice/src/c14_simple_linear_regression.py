'''
Created on Jul 25, 2017

@author: loanvo
'''
from c05_statistics import correlation, standard_variation, mean, de_mean
from c08_gradient_descent import minimize_stochastic
import random

def predict(alpha, beta, x_i):
    return alpha + beta * x_i

def error(alpha, beta, x_i, y_i):
    """the error from predicting beta * x_i + alpha when the actual value is y_i"""
    return y_i - predict(alpha, beta, x_i)

def sum_of_squared_errors(alpha, beta, x, y):
    return sum([error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x,y)])


def least_squares_fit(x, y):
    """Given training values for x and y, find the least-squares values of alpha and beta 
    which minimize sum of square of errors. 
    This codes uses equations derived generally from linear algebra to calculate (alpha, beta)
    """
    beta = correlation(x,y) * standard_variation(y) / standard_variation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta

def total_sum_of_squares(y):
    """the total squared variation of y_i's from their mean"""
    return sum(y_i ** 2 for y_i in de_mean(y))

def r_squared(alpha, beta, x, y):
    """the fraction of variation in y captured by the model, which equals
    1 - the fraction of variation in y not captured by the model"""
    return 1.0 -  sum_of_squared_errors(alpha, beta, x, y)/total_sum_of_squares(y)



def squared_error_i(x_i_y_i, theta):
    x_i, y_i = x_i_y_i
    alpha, beta = theta
    return error(alpha, beta, x_i, y_i) ** 2

def squared_error_gradient_i(x_i_y_i, theta):
    x_i, y_i = x_i_y_i
    alpha, beta = theta
    return [-2 * error(alpha, beta, x_i, y_i),
            -2 * x_i * error(alpha, beta, x_i, y_i), ]

def least_squares_fit_with_stochastic_gradient_decent(x,y):
    """Given training values for x and y, find the least-squares values of alpha and beta 
    which minimize sum of square of errors. 
    This codes uses stochastic gradient descent to search for the optimal (alpha, beta)
    """
    # choose random value to start
    random.seed(0)
    theta_0 = [random.random(), random.random()]
    data = list(zip(x,y))
    alpha, beta = minimize_stochastic(squared_error_i,
                                      squared_error_gradient_i,
                                      data,
                                      theta_0
                                      )
    return alpha, beta