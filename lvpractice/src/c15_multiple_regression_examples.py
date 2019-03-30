'''
Created on Jul 26, 2017

@author: loanvo
'''

import random
from c15_multiple_regression import estimate_beta, bootstrap_statistic, estimate_sample_beta, \
    p_value, estimate_beta_ridge, multiple_r_squared
from c05_statistics import median, standard_variation
from c04_linear_algebra import augmented_matrix, dot_product

# data for multiple-regression examples
x = [[1,49,4,0],[1,41,9,0],[1,40,8,0],[1,25,6,0],[1,21,1,0],[1,21,0,0],[1,19,3,0],[1,19,0,0],[1,18,9,0],[1,18,8,0],[1,16,4,0],[1,15,3,0],[1,15,0,0],[1,15,2,0],[1,15,7,0],[1,14,0,0],[1,14,1,0],[1,13,1,0],[1,13,7,0],[1,13,4,0],[1,13,2,0],[1,12,5,0],[1,12,0,0],[1,11,9,0],[1,10,9,0],[1,10,1,0],[1,10,1,0],[1,10,7,0],[1,10,9,0],[1,10,1,0],[1,10,6,0],[1,10,6,0],[1,10,8,0],[1,10,10,0],[1,10,6,0],[1,10,0,0],[1,10,5,0],[1,10,3,0],[1,10,4,0],[1,9,9,0],[1,9,9,0],[1,9,0,0],[1,9,0,0],[1,9,6,0],[1,9,10,0],[1,9,8,0],[1,9,5,0],[1,9,2,0],[1,9,9,0],[1,9,10,0],[1,9,7,0],[1,9,2,0],[1,9,0,0],[1,9,4,0],[1,9,6,0],[1,9,4,0],[1,9,7,0],[1,8,3,0],[1,8,2,0],[1,8,4,0],[1,8,9,0],[1,8,2,0],[1,8,3,0],[1,8,5,0],[1,8,8,0],[1,8,0,0],[1,8,9,0],[1,8,10,0],[1,8,5,0],[1,8,5,0],[1,7,5,0],[1,7,5,0],[1,7,0,0],[1,7,2,0],[1,7,8,0],[1,7,10,0],[1,7,5,0],[1,7,3,0],[1,7,3,0],[1,7,6,0],[1,7,7,0],[1,7,7,0],[1,7,9,0],[1,7,3,0],[1,7,8,0],[1,6,4,0],[1,6,6,0],[1,6,4,0],[1,6,9,0],[1,6,0,0],[1,6,1,0],[1,6,4,0],[1,6,1,0],[1,6,0,0],[1,6,7,0],[1,6,0,0],[1,6,8,0],[1,6,4,0],[1,6,2,1],[1,6,1,1],[1,6,3,1],[1,6,6,1],[1,6,4,1],[1,6,4,1],[1,6,1,1],[1,6,3,1],[1,6,4,1],[1,5,1,1],[1,5,9,1],[1,5,4,1],[1,5,6,1],[1,5,4,1],[1,5,4,1],[1,5,10,1],[1,5,5,1],[1,5,2,1],[1,5,4,1],[1,5,4,1],[1,5,9,1],[1,5,3,1],[1,5,10,1],[1,5,2,1],[1,5,2,1],[1,5,9,1],[1,4,8,1],[1,4,6,1],[1,4,0,1],[1,4,10,1],[1,4,5,1],[1,4,10,1],[1,4,9,1],[1,4,1,1],[1,4,4,1],[1,4,4,1],[1,4,0,1],[1,4,3,1],[1,4,1,1],[1,4,3,1],[1,4,2,1],[1,4,4,1],[1,4,4,1],[1,4,8,1],[1,4,2,1],[1,4,4,1],[1,3,2,1],[1,3,6,1],[1,3,4,1],[1,3,7,1],[1,3,4,1],[1,3,1,1],[1,3,10,1],[1,3,3,1],[1,3,4,1],[1,3,7,1],[1,3,5,1],[1,3,6,1],[1,3,1,1],[1,3,6,1],[1,3,10,1],[1,3,2,1],[1,3,4,1],[1,3,2,1],[1,3,1,1],[1,3,5,1],[1,2,4,1],[1,2,2,1],[1,2,8,1],[1,2,3,1],[1,2,1,1],[1,2,9,1],[1,2,10,1],[1,2,9,1],[1,2,4,1],[1,2,5,1],[1,2,0,1],[1,2,9,1],[1,2,9,1],[1,2,0,1],[1,2,1,1],[1,2,1,1],[1,2,4,1],[1,1,0,1],[1,1,2,1],[1,1,2,1],[1,1,5,1],[1,1,3,1],[1,1,10,1],[1,1,6,1],[1,1,0,1],[1,1,8,1],[1,1,6,1],[1,1,4,1],[1,1,9,1],[1,1,9,1],[1,1,4,1],[1,1,2,1],[1,1,9,1],[1,1,0,1],[1,1,8,1],[1,1,6,1],[1,1,1,1],[1,1,1,1],[1,1,5,1]]
daily_minutes_good = [68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]
x_daily_minutes_good = augmented_matrix(x, daily_minutes_good)

# estimate betas in multiple-regression model (using gradient descent)
random.seed(0)
betas = estimate_beta(x_daily_minutes_good)
print("beta = ", betas)

# Bootstrap
# Generate 101 points all very close to 100
close_to_100 = [99.5 + random.random() for _ in range(100)]
# Generate 101 points including 50 of them near 0, 50 of them near 200 and 1 of them near to 100
far_from_100 = [random.random() for _ in range(50)] + \
    [200 + random.random() for _ in range(50)] + \
    [99.5 + random.random()]
# if we compute median of each in a typical way:
print("\n\nClose_to_100 has a median: ", median(close_to_100))
print("Far_from_100 has a median: ", median(far_from_100))
# if we do bootstrap:
close_to_100_bootstrap_median = bootstrap_statistic(close_to_100, median, 100)
far_from_100_bootstrap_median = bootstrap_statistic(far_from_100, median, 100)
print("With bootstrap, close_to_100 has a median: ", close_to_100_bootstrap_median)
print("And its standard deviation is ", standard_variation(close_to_100_bootstrap_median))
print("With bootstrap, far_from_100 has a median: ", far_from_100_bootstrap_median)
print("And its standard deviation is ", standard_variation(far_from_100_bootstrap_median))
# estimate beta with bootstrap
random.seed(0) 
bootstrap_betas = bootstrap_statistic(x_daily_minutes_good, estimate_sample_beta, 100)

# standard deviation of each coefficient
bootstrap_standard_errors = [standard_variation(
    [bootstrap_beta[j] for bootstrap_beta in bootstrap_betas]) for j in range(4)]
print("\n\nStandard deviation of bootstrap betas:")    
print(*bootstrap_standard_errors, sep = "\n")
ps = [p_value(beta, bootstrap_standard_error) 
      for beta, bootstrap_standard_error in zip(betas, bootstrap_standard_errors)]
print("\n\nP-values of beta coefficients:", ps)

# Ridge regularization
# if alpha = 0.0 --> no ridge regularization
random.seed(0)
beta_0 = estimate_beta_ridge(x_daily_minutes_good, alpha=0.0)
print("\n\nAs alpha = 0, beta = ", beta_0)
print("Beta's length: ", dot_product(beta_0[1:],beta_0[1:]))
print("Multiple r-squared:", multiple_r_squared(x, daily_minutes_good, beta_0))
# As we increase alpha, the goodness of fit gets worse, but the size of beta gets smaller
# alpha = .01
random.seed(0)
beta_0_01 = estimate_beta_ridge(x_daily_minutes_good, alpha=0.01)
print("\n\nAs alpha = 0.01, beta = ", beta_0_01)
print("Beta's length: ", dot_product(beta_0_01[1:],beta_0_01[1:]))
print("Multiple r-squared:", multiple_r_squared(x, daily_minutes_good, beta_0_01))
# alpha = .1
random.seed(0)
beta_0_1 = estimate_beta_ridge(x_daily_minutes_good, alpha=0.1)
print("\n\nAs alpha = 0.1, beta = ", beta_0_1)
print("Beta's length: ", dot_product(beta_0_1[1:],beta_0_1[1:]))
print("Multiple r-squared:", multiple_r_squared(x, daily_minutes_good, beta_0_1))
# alpha = 1
random.seed(0)
beta_1 = estimate_beta_ridge(x_daily_minutes_good, alpha=1.0)
print("\n\nAs alpha = 1, beta = ", beta_1)
print("Beta's length: ", dot_product(beta_1[1:],beta_1[1:]))
print("Multiple r-squared:", multiple_r_squared(x, daily_minutes_good, beta_1))
# alpha = 10
random.seed(0)
beta_10 = estimate_beta_ridge(x_daily_minutes_good, alpha=10.0)
print("\n\nAs alpha = 10, beta = ", beta_10)
print("Beta's length: ", dot_product(beta_10[1:],beta_10[1:]))
print("Multiple r-squared:", multiple_r_squared(x, daily_minutes_good, beta_10))


