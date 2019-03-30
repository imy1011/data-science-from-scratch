'''
Created on July 2, 2017

@author: loanvo
'''

import math
import random

def uniform_pdf(x):
    return 1 if x >= 0 and x < 1 else 0

def uniform_cdf(x):
    if x < 0:
        return 0
    if x<=1:
        return x
    else:
        return 1
    
def normal_pdf(x, mu=0, sigma=1):
    return math.exp(- (x - mu) ** 2 / (2 * (sigma ** 2))) / (math.sqrt(2 * math.pi) * sigma ) 

def normal_cdf(x, mu=0, sigma=1):
    return (1 + math.erf((x-mu) / math.sqrt(2) / sigma)) / 2

def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    """find approximate inverse using binary search"""        
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    
    low_z = -10 # normal_pdf(-10) ~ 0
    high_z = 10 # normal_pdf(-10) ~ 1
    while high_z - low_z > tolerance:
        mid_z = (low_z + high_z)/2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            low_z = mid_z
        elif mid_p > p:
            high_z = mid_z
        else:
            break
    return mid_z      

def bernouli_trial(p):     
    """ P(X=1) = p and P(X=0) = 1-p"""
    return 1 if random.random() < p else 0

def binomial(n, p):
    return sum(bernouli_trial(p) for _ in range(n))
                         