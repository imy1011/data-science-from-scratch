'''
Created on July 2, 2017

@author: loanvo
'''

import random

def sum_of_squares(theta):
    return sum([theta_i**2 for theta_i in theta])

def difference_quotient(f, theta, h): #derivative of f(theta): limit of the difference quotient as h approaches 0
    return (f(theta+h)-f(theta))/h

def square(theta):
    return theta*theta

def derivative(theta): #the exact form/calculation of derivative_of_square theta^2
    return 2*theta



'''
Using the gradient: 
    Let consider f(theta) = sum(theta_i^2)
    Goal: find the minimum point of the function using gradient descent (choose your step_size and tolerance)
    Loan's note: in a learning problem, f(theta) is usually the emperical risk and theta is the learning parameters.
                Usually, f is also a function of observed/unobserved data (x,y).
                In this example, we simplify the function f and let it depends on theta only (independent of (x,y))
'''

# gradient of the function sum_of_squares f(theta) = sum(theta_i^2)
def sum_of_square_gradient(theta):
    return [ 2 * theta_i for theta_i in theta]

# the next searching point would be the sum of the previous vector and the scaled (by step_size value) gradient vector
def step(theta, df, step_size): # theta: starting value, df: gradient of the function at the starting value
    return [ theta_i + step_size*dfi for theta_i, dfi in zip(theta, df) ]

# distance between two consecutive values of target_fn: whether there is improvement in minimum value.
def distance(next_theta, theta, target_fn):
    return abs(target_fn(next_theta)-target_fn(theta))# calculate the distance between function values at two points


'''
Depending on the chosen step size, certain step sizes might result in invalid inputs for target function.
--> we will need to create a "safe apply" function that returns infinity for invalid inputs.
'''
def safe(f):
    def safe_f(*args, **kargs):
        try:
            return f(*args, **kargs)
        except:
            return float('inf')
    return safe_f

'''
General Problem solved by gradient descent:
- Given: target_fn (some error in a model), its gradient_fn
- Goal: want to find the parameters that minimizes the error function target_fn
- Choose: starting value for the parameters theta_0
'''

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance = 0.000001):

    # there is no rigorous way for choosing the right step size. Common method: use a fixed sep size,
    # gradually shrinking the step size over time, or at each step, choosing the step size that minimizes
    # the value of the objective function.
    # Here we start with a set of step_sizes, but at each gradient-descent-step we choose the step_size that
    # minimizes target_fn
    step_sizes = [100, 10, 1, .1, .01, .001, .0001, .00001]
    #step_sizes = [10, 1, .1, .01, .001, .0001, .00001]
    
    theta = theta_0
    target_fn = safe(target_fn)
    value = target_fn(theta)
    #i=0
    while True:
        '''
        i += 1
        print(i,":",next_theta, "and", next_value)
        if i%10 == 0:
            input("Pause!")
        '''
        gradientDirection = gradient_fn(theta)
        # for each step size, we have a next_theta
        next_thetas = [ step(theta, gradientDirection, -step_size) for step_size in step_sizes ]
        # choose the next_theta that minimize the target_fn
        next_theta = min(next_thetas, key = target_fn)
        next_value = target_fn(next_theta)
        
        # step if we're 'converging'
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value
        


'''
To maximize a function, we can minimize its negative
'''
def negate(f):
    return lambda *args, **kargs: -f(*args, **kargs)
def negate_all(f):
    return lambda *args, **kargs: [-f_i for f_i in f(*args, **kargs)]
def maximize_batch(target_fn, gradient_fn, theta_0, tolerance = 0.000001):
    return minimize_batch(negate(target_fn), 
                          negate_all(gradient_fn), 
                          theta_0, tolerance)

'''
Stochastic Gradient Descent
https://en.wikipedia.org/wiki/Stochastic_gradient_descent
https://en.wikipedia.org/wiki/Empirical_risk_minimization

In supervised learning problem, we have to learn the hypothesis function h: X-->Y
which outputs an object y in Y for each given x in X. 
To do so we have our training set (x1, y1),...(xm, ym) and assume that there is a joint
probability distribution P(x,y) over X and Y, i.e., the training set consists of m instances
drawn i.i.d. from P(x,y).
The assumption of a joint probability distribution allow us to model uncertainty in predictions (e.f. from noise in data)

Given the non-negative loss function L(yhat, y) (note: yhat=h(x, theta)) which measures how different the
prediction yhat of a hypothesis is from the true output y.
The risk associated with hypthesis h(x, theta) is the expection of the loss function:
R(h) = E[L(h(x),y)]
Nevertheless R(h) CANNOT be computed because the distribution P(x,y) is unknown
--> we only comute an approximation, called empirical risk: averaging the loss function on the training set:

R_empirical(h) = 1/m sum( L(h(xi),yi) )

--> Our goal is choosing a hypothesis h_hat which minimizes the empirical risk
--> finding the optimal hypothesis h(x, theta), i.e., finding its parameter theta, which minimizes the empirical risk,
i.e., the target function of our optimization problem is  the empirical risk 
which is additive R_empirical(h) = 1/m sum( L(h(xi),yi) ): the predictive error on the whole data set 
is simply the sum of the predictive errors for each data point.

--> STOCHASTIC GRADIENT DESCENT: 
R_empirical(theta) = 1/m sum (R_i(theta))
If using the batch gradient descent minimize_batch, we would need to perform the following iterations:

theta := theta - step_size * R_empirical_gradient(theta) = theta - step_size * sum(R_i_gradient(theta)) / m

When m (number of samples in training set) is enormous and R_i_gradient doesn't have a simple form,
the evaluation of sums of gradients becomes very expensive -->  To economize on the computational cost at every iteration, 
stochastic gradient descent samples a subset of summand functions at every step. 
This is very effective in the case of large-scale machine learning problems

In stochastic (or "on-line") gradient descent, the true gradient of R_empirical is approximated by a gradient at a single example
theta := theta - step_size * R_i_gradient(theta) . As the algorithm sweeps through the trainign set, it performs this update for each
training sample. Several passes can be made over the training set until the algorithm converges (the data should be shuffled for 
each pass to prevent cycles)

In pseudocode, stochastic gradient descent can be presented as follows:
    1. Choose an initial vector of parameters theta and learning rate, i.e. step_size
    2. Repeat until an approximate minimum is obtained:
        - Randomly shuffle examples in the training set.
        - For i = 1, 2,..., m, do: 
            theta := theta - step_size * R_i_gradient(theta)


'''
# Loan's note: on page 100 of Joel Grus's book, the author wrote in_random_order with input as a "zip" object
# However the last line in that function (yield data[i]) won't work in python 3 as data is a zip object and so 
# it is not accessible by the [index]
def in_random_order(data): #function to shuffle samples for each pass
    indexes = list(range(len(data)))
    random.shuffle(indexes)
    for i in indexes:
        yield data[i]

def minimize_stochastic(target_fn, gradient_fn, data, theta_0, 
                        alpha_0 = .0001, tolerance = .000001): # Add tolerance
    
    target_fn = safe(target_fn)
    theta = theta_0 # initial guess
    alpha = alpha_0 # initial step sizes
    min_theta, min_value = None, float("inf") # the minimum so far
    iterations_with_no_improvement = 0
    #i = 0
    while iterations_with_no_improvement < 100: # if we ever go 100 iterations with no improvement, stop
        value = sum([ target_fn(data_i, theta) for data_i in data ])
        # Loan added the term tolerance here to avoid 'infinite' while-loop
        # i.e. stop finding minimum if value doesn't change much.
        if (value + tolerance) < min_value: #if we've found a new minimum
            min_theta, min_value = theta, value #remember the new minimum and the corresponding theta
            alpha = alpha_0 #we also reset alpha to the original/initial step size
            iterations_with_no_improvement = 0 # and also reset iterations_with_no_improvement
        else:    
            iterations_with_no_improvement += 1
            alpha *=.9 #try shrinking the step size if there is no improvement    
        # and take a gradient step for each of the data points    
        for data_i in in_random_order(data):
            theta = step(theta, gradient_fn(data_i, theta), -alpha)
        #i += 1
        #print(i,":", theta, "and", value)      
    return min_theta

def maximize_stochastic(target_fn, gradient_fn, data, theta_0, alpha_0 = .01, tolerance = .000001):
    return minimize_stochastic(negate(target_fn), 
                               negate_all(gradient_fn),
                               data, 
                               theta_0,
                               alpha_0,
                               tolerance)
