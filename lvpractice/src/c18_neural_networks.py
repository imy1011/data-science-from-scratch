'''
Created on Jul 30, 2017

@author: loanvo
'''

import math
from matplotlib import pyplot as plt
import matplotlib 
from c04_linear_algebra import dot_product


def step_function(x):
    return 0 if x<0 else 1

def perceptron_output(weights, bias, x):
    """returns 1 if the perceptron 'fires', 0 if not"""
    return step_function(dot_product(weights,x) + bias)

def sigmoid(t):
    """
    The term sigmoid and logistic (function) are used interchangeably
    Technically “sigmoid” refers to the shape of the function, 
    “logistic” to this particular function"""
    return 1/(1+math.exp(-t))

def neuron_output(weights, inputs):
    """
    Weights: includes the bias to its end 
    Inputs: Each neuron also given a bias input that always equals 1
    """
    return sigmoid(dot_product(weights, inputs))

def feed_forward(neural_network, input_vector):
    """takes in a neural network (represented as a list of lists of lists of weights)
    and returns the output from forward-propagating the input
    outputs: list of outputs of all hidden layers and output layer
    """
    outputs = []
    for layer in neural_network:
        layer_input = input_vector + [1] #adding the bias input
        output = [neuron_output(neuron, layer_input) for neuron in layer ] # list of output of each neuron in the layer
        outputs.append(output) # save the output of this neural layer
        input_vector = output # input_vector of the next layer
    return outputs

def predict_with_neural_network(neural_network, input_vector):
    return feed_forward(neural_network, input_vector)[-1]

def backpropagate(network, input_vector, targets):   
    """
    Our network has some set of weights. We then adjust the weights using the following algorithm:
    
    - Run feed_forward on an input vector to produce the outputs of all the neurons in the network.
    - This results in an error for each output neuron, i.e., the difference between its output 
    and its target.
    - Compute the gradient of this error as a function of the neuron’s weights, and adjust its 
    weights in the direction that most decreases the error.
    - “Propagate” these output errors backward to infer errors for the hidden layer.
    - Compute the gradients of these errors and adjust the hidden layer’s weights in the same manner.

    Typically we run this algorithm many times for our entire training set until the network converges
    """
    hidden_neurons = network[0]
    output_neurons = network[-1] 
    
    # Calculate the output given input_vector input
    all_outputs = feed_forward(network, input_vector)
    hidden_outputs = all_outputs[0]
    outputs = all_outputs[-1]
    output_inputs = hidden_outputs #input of the output layer is the output of the hidden layer right before it
    
    # Error at output neuron i (denote its input as x):
    # square_output_error_i (denoted as soe) = (outputs[i] - targets[i])) ** 2
    #                                        = (sigmoid(dot(wi,x)) - targets[i]) ** 2 (where x is the input of this layer)
    # gradient_of_square_output_error = [d_soe/d_wi1, d_soe/d_wi2, ..., d_soe/win]
    # where d_soe/d_wij : partial derivative of square_output_error (at output neuron i) with respect to wij
    # i.e., d_soe/d_wij = 2*output_error_i*(1-outputs[i])*outputs[i]*x_j
    
    # Loan's note: I am not sure why Joel Grus doesn't include the constant "2"
    # output_deltas = [2 * (1-output) * output * (output - target) for output, target in zip(outputs, targets)]
    output_deltas = [(1-output) * output * (output - target) for output, target in zip(outputs, targets)]

    # Adjust the weight vector of each neuron in the output layer in the negative direction of 
    # the gradient of the error function (a function of weight vector elements)
    for i, output_neuron in enumerate(output_neurons): # working on the ith output neuron
        for j, output_input in enumerate(output_inputs + [1]): #input vector includes bias input 1
            partial_grad_with_respect_to_weight_j = output_deltas[i] * output_input
            # adjust the weight_j (of neuron i) with the partial gradient
            output_neuron[j] -= partial_grad_with_respect_to_weight_j
        
    # Propagate output error to infer errors in the hidden layer
    # Again, the partial derivative of square of error with respect to weight j (of hidden neuron i):
    # d_soe/d_wij = 2*hidden_output_error_i*(1-hidden_outputs[i])*hidden_outputs[i]*hidden_input_j
    # Nevertheless, we don't have "hidden_output_error" (as we don't have a desired target for 
    # each hidden layer).
    # We replace hidden_output_errors by back-propagating the network output errors into this stage.
    # Particularly, we project output_deltas (derived from the network output_errors) into this stage
    hidden_deltas = [(1-hidden_output) * hidden_output * 
                     dot_product(output_deltas, [output_neuron[i] for output_neuron in output_neurons]) 
                     for i, hidden_output in enumerate(hidden_outputs)]
        
    # adjusting weights of hidden_neuron
    for i, hidden_neuron in enumerate(hidden_neurons):
        for j, hidden_input in enumerate(input_vector + [1]):
            hidden_pseduo_gradient_j = hidden_deltas[i] * hidden_input 
            hidden_neuron[j] -= hidden_pseduo_gradient_j

def patch(x, y, hatch, color):
    """return a matplotlib 'patch' object with the specified
    location, crosshatch pattern, and color"""
    return matplotlib.patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                        hatch=hatch, fill=False, color=color)
               
def show_weights(network, layer_idx, neuron_idx):
    weights = network[layer_idx][neuron_idx]
    print(weights)
    abs_weights = [abs(weight) for weight in weights]
    weights_in_grid_arrangment = [abs_weights[idx:idx+5] for idx in range(0,25,5)]
    ax = plt.gca() ## to use hatching, we'll need the axis
    ax.imshow(weights_in_grid_arrangment, # here same as plt.imshow
              cmap=matplotlib.cm.get_cmap("binary") , # use white-black color scale
              interpolation='nearest') # plot blocks as blocks
    # cross-hatch the negative weights
    for i in range(5): # row
        for j in range(5): # column
            if weights[5*i + j] < 0: # row i, column j = weights[5*i + j]
                # add black and white hatches, so visible whether dark or light
                ax.add_patch(patch(j, i, '/', "white"))
                ax.add_patch(patch(j, i, '\\', "black"))
