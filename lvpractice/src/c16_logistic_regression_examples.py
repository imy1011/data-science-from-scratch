'''
Created on Jul 26, 2017

@author: loanvo
'''
import random
from functools import partial
from matplotlib import pyplot as plt
from c11_machine_learning import train_test_split, precision, recall
from c10_working_with_data import rescale
from c08_gradient_descent import maximize_batch, maximize_stochastic
from c16_logistic_regression import logistic_log_likelihood, logistic_log_gradient, \
    logistic_log_likelihood_i, logistic_log_gradient_i, logistic
from c04_linear_algebra import augmented_matrix, dot_product
from c15_multiple_regression import estimate_beta

"""
Preparing the data
"""
# Preparing the data
data = [(0.7,48000,1),(1.9,48000,0),(2.5,60000,1),(4.2,63000,0),(6,76000,0),(6.5,69000,0),(7.5,76000,0),(8.1,88000,0),(8.7,83000,1),(10,83000,1),(0.8,43000,0),(1.8,60000,0),(10,79000,1),(6.1,76000,0),(1.4,50000,0),(9.1,92000,0),(5.8,75000,0),(5.2,69000,0),(1,56000,0),(6,67000,0),(4.9,74000,0),(6.4,63000,1),(6.2,82000,0),(3.3,58000,0),(9.3,90000,1),(5.5,57000,1),(9.1,102000,0),(2.4,54000,0),(8.2,65000,1),(5.3,82000,0),(9.8,107000,0),(1.8,64000,0),(0.6,46000,1),(0.8,48000,0),(8.6,84000,1),(0.6,45000,0),(0.5,30000,1),(7.3,89000,0),(2.5,48000,1),(5.6,76000,0),(7.4,77000,0),(2.7,56000,0),(0.7,48000,0),(1.2,42000,0),(0.2,32000,1),(4.7,56000,1),(2.8,44000,1),(7.6,78000,0),(1.1,63000,0),(8,79000,1),(2.7,56000,0),(6,52000,1),(4.6,56000,0),(2.5,51000,0),(5.7,71000,0),(2.9,65000,0),(1.1,33000,1),(3,62000,0),(4,71000,0),(2.4,61000,0),(7.5,75000,0),(9.7,81000,1),(3.2,62000,0),(7.9,88000,0),(4.7,44000,1),(2.5,55000,0),(1.6,41000,0),(6.7,64000,1),(6.9,66000,1),(7.9,78000,1),(8.1,102000,0),(5.3,48000,1),(8.5,66000,1),(0.2,56000,0),(6,69000,0),(7.5,77000,0),(8,86000,0),(4.4,68000,0),(4.9,75000,0),(1.5,60000,0),(2.2,50000,0),(3.4,49000,1),(4.2,70000,0),(7.7,98000,0),(8.2,85000,0),(5.4,88000,0),(0.1,46000,0),(1.5,37000,0),(6.3,86000,0),(3.7,57000,0),(8.4,85000,0),(2,42000,0),(5.8,69000,1),(2.7,64000,0),(3.1,63000,0),(1.9,48000,0),(10,72000,1),(0.2,45000,0),(8.6,95000,0),(1.5,64000,0),(9.8,95000,0),(5.3,65000,0),(7.5,80000,0),(9.9,91000,0),(9.7,50000,1),(2.8,68000,0),(3.6,58000,0),(3.9,74000,0),(4.4,76000,0),(2.5,49000,0),(7.2,81000,0),(5.2,60000,1),(2.4,62000,0),(8.9,94000,0),(2.4,63000,0),(6.8,69000,1),(6.5,77000,0),(7,86000,0),(9.4,94000,0),(7.8,72000,1),(0.2,53000,0),(10,97000,0),(5.5,65000,0),(7.7,71000,1),(8.1,66000,1),(9.8,91000,0),(8,84000,0),(2.7,55000,0),(2.8,62000,0),(9.4,79000,0),(2.5,57000,0),(7.4,70000,1),(2.1,47000,0),(5.3,62000,1),(6.3,79000,0),(6.8,58000,1),(5.7,80000,0),(2.2,61000,0),(4.8,62000,0),(3.7,64000,0),(4.1,85000,0),(2.3,51000,0),(3.5,58000,0),(0.9,43000,0),(0.9,54000,0),(4.5,74000,0),(6.5,55000,1),(4.1,41000,1),(7.1,73000,0),(1.1,66000,0),(9.1,81000,1),(8,69000,1),(7.3,72000,1),(3.3,50000,0),(3.9,58000,0),(2.6,49000,0),(1.6,78000,0),(0.7,56000,0),(2.1,36000,1),(7.5,90000,0),(4.8,59000,1),(8.9,95000,0),(6.2,72000,0),(6.3,63000,0),(9.1,100000,0),(7.3,61000,1),(5.6,74000,0),(0.5,66000,0),(1.1,59000,0),(5.1,61000,0),(6.2,70000,0),(6.6,56000,1),(6.3,76000,0),(6.5,78000,0),(5.1,59000,0),(9.5,74000,1),(4.5,64000,0),(2,54000,0),(1,52000,0),(4,69000,0),(6.5,76000,0),(3,60000,0),(4.5,63000,0),(7.8,70000,0),(3.9,60000,1),(0.8,51000,0),(4.2,78000,0),(1.1,54000,0),(6.2,60000,0),(2.9,59000,0),(2.1,52000,0),(8.2,87000,0),(4.8,73000,0),(2.2,42000,1),(9.1,98000,0),(6.5,84000,0),(6.9,73000,0),(5.1,72000,0),(9.1,69000,1),(9.8,79000,1),]
data = list(map(list, data))
x = [[1] + data_i[:-1] for data_i in data] # each x_i is (1, experience, salary)
rescaled_x = rescale(x)
y = [data_i[-1] for data_i in data]
# Dividing data into training and testing set:
random.seed(0)
x_train, x_test, y_train, y_test = train_test_split(rescaled_x, y, .33)
x_train_y_train = augmented_matrix(x_train, y_train)


"""
Linear regression:
"""
print("\n\nLinear regression:")
beta_hat_linear = estimate_beta(x, y) #estimate_beta(x_train, y_train)
print("Beta estimated from linear regression:", beta_hat_linear)

"""
Logistic regression:
"""
print("\n\nLogistic regression:")
beta_0 = [random.random() for _ in x_train[0]] 
# Want to maximize logistic_log_likelihood with batch gradient descent
beta_hat_logistic_batch = maximize_batch(partial(logistic_log_likelihood, x_train, y_train),
                                         partial(logistic_log_gradient, x_train, y_train),
                                         beta_0)
print("Beta estimated from logistic regression model with batch optimization:", 
      beta_hat_logistic_batch)
# Want to maximize logistic_log_likelihood with stochastic gradient descent
beta_hat_logistic_sgd = maximize_stochastic(logistic_log_likelihood_i,
                                            logistic_log_gradient_i,
                                            x_train_y_train,
                                            beta_0)
print("Beta estimated from logistic regression model with stochastic optimization:", 
      beta_hat_logistic_sgd)

# Calculate precision and recall:
tp, fp, tn, fn = 0, 0, 0, 0
for x_test_i, y_test_i in zip(x_test, y_test):
    predict_label = 1 if logistic(dot_product(x_test_i, beta_hat_logistic_batch)) > 0.5 else 0
    if y_test_i == 1:
        if predict_label == 1:
            tp += 1
        else:
            fn += 1
    else:
        if predict_label == 1:
            fp += 1
        else:
            tn += 1
print("\n\nPrecision of our logistic regression:", precision(tp, fp))
print("Recall of our logistic regression:", recall(tp, fn))


prediction_probability = [logistic(dot_product(x_test_i, beta_hat_logistic_batch)) for x_test_i in x_test]
right_classification = list(filter(lambda z: (z[0] >= .5) ==  z[1], zip(prediction_probability, y_test)))
wrong_classification = list(filter(lambda z: (z[0] >= .5) !=  z[1], zip(prediction_probability, y_test)))
plt.figure()
plt.scatter([rc[0] for rc in right_classification], [rc[1] for rc in right_classification], \
            color = 'red', marker = '+', label = "Correct classification")
plt.scatter([wc[0] for wc in wrong_classification], [wc[1] for wc in wrong_classification], \
            color = 'blue', marker = '.', label = "Wrong classification")
plt.xlabel("predicted probability")
plt.ylabel("actual outcome")
plt.legend(loc=5)
plt.title("Logistic Regression Predicted vs. Actual")
plt.show() 
