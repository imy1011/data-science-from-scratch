'''
Created on Jul 19, 2017
By Loan Vo
Original File Path: /Users/loanvo/GitHub/py/joelgrus/joelgrus/src/c08_gradient_descent_examples.py

Descriptions:
- Goal:
- Input:
- Output:
'''
from functools import partial
import numpy as np
import random
from c08_gradient_descent import difference_quotient, square, derivative, \
    sum_of_square_gradient, step, distance, sum_of_squares, minimize_batch
from matplotlib import pyplot as plt

'''
Comparing actual derivatives and estimated ones
'''
#
derivative_estimate = partial(difference_quotient,square,h=.00001)  #estimation for derivative_of_square theta^2
#
x = np.arange(-10,10)
plt.figure()
plt.plot(x, list(map(derivative, x)), 'rx', label = "Actual")
plt.plot(x, list(map(derivative_estimate, x)), 'b+', label = "Estimate")
plt.legend(loc=9)
plt.title("Actual Derivatives vs Estimates")


'''
Find parameter that minimizes a cost function with GRADIENT DESCENT
'''
# Randomly pick a staring point
theta = [ random.randint(-10,10) for _ in range(3) ]

#thetas = [ theta ]
# if the two consecutive points are very close to each other, i.e. their distance < tolerance, then stop the search
tolerance = .00000001

# calculate the next point: on the gradient direction
step_size = -.01 # take a negative gradient step

while True:
    next_theta = step(theta, sum_of_square_gradient(theta), step_size)
    #thetas.append(next_theta)
    if distance(next_theta, theta, sum_of_squares) < tolerance: #stop if we're converging
        break
    theta = next_theta
    
#print("Optimal theta (an example of applying Gradient Descent):", thetas[-1])
#print("Minimum value (an example of applying Gradient Descent):", sum_of_squares(thetas[-1]))
#print(distance(thetas[-1],thetas[-2],sum_of_squares), distance(thetas[-1],thetas[-2],sum_of_squares)<tolerance )



#Calling the minimize_batch function and display the output
theta_0_minimize_batch = [ random.randint(-10,10) for _ in range(3) ]
theta_minimize_batch = minimize_batch(sum_of_squares, sum_of_square_gradient, theta_0_minimize_batch)
print('theta_minimize_batch:', theta_minimize_batch)
print('minimum_value_minimize_batch:', sum_of_squares(theta_minimize_batch))


plt.show()

