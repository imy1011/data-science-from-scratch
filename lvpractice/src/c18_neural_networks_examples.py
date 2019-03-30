'''
Created on Jul 30, 2017

@author: loanvo
'''
from c18_neural_networks import feed_forward, backpropagate, predict_with_neural_network, show_weights
import random
from matplotlib import pyplot as plt


xor_network = [# hidden layer
               [[20, 20, -30],  #and gate
                [20, 20, -10]], # or gate
               # output layer
               [[-60, 60, -30]]]  # 2nd input but not 1st input

print("\nThe xor_network:")
for x in [0, 1]:
    for y in [0, 1]:
        # feed_forward produces the outputs of every neuron
        # feed_forward[-1] is the outputs of the output-layer neurons
        print("Input:", x, y, "--> Output:",feed_forward(xor_network, [x, y])[-1])
        

"""
Testing the back propagation algorithm in building the neuron network
"""

""" Preparing the data """
training_raw_digits = [
          """11111
             1...1
             1...1
             1...1
             11111""",

          """..1..
             ..1..
             ..1..
             ..1..
             ..1..""",

          """11111
             ....1
             11111
             1....
             11111""",

          """11111
             ....1
             11111
             ....1
             11111""",

          """1...1
             1...1
             11111
             ....1
             ....1""",

          """11111
             1....
             11111
             ....1
             11111""",

          """11111
             1....
             11111
             1...1
             11111""",

          """11111
             ....1
             ....1
             ....1
             ....1""",

          """11111
             1...1
             11111
             1...1
             11111""",

          """11111
             1...1
             11111
             ....1
             11111"""]
testing_raw_digits = [
                    """
                    .111.
                    ...11
                    ..11.
                    ...11
                    .111.
                    """,
                    """
                    .111.
                    1..11
                    .111.
                    1..11
                    .111.
                    """
                    ]
def make_digit(raw_digits):  
    return [[1 if c == "1" else 0 
            for raw_line in raw_digit.splitlines() for c in raw_line.strip() if raw_line.strip()] 
            for raw_digit in raw_digits]

training_inputs = make_digit(training_raw_digits)
testing_inputs = make_digit(testing_raw_digits)

""" Initialize """
random.seed(0) # so that we will get the repeat result for every run   
input_size = len(training_inputs[0])
num_neuron_in_a_hidden_layer = 5 # there is 5 neurons in the hidden layer (assume there is only ONE hidden layer)
output_size = len(training_inputs) # we need 10 outputs for each input (if input is digit 9, then the 9th output=1 and the others = 0)

# The desired/targeted outputs for each input
targets = [[1 if output == digit else 0 for output in range(output_size) ] for digit in range(output_size)]

# Initialize weight vector for each neuron in the hidden layer
# each hidden neuron has one weight per input, plus a bias weight
hidden_layer = [[random.random() 
                 for _ in range(input_size)]    # each hidden neuron has input_size inputs
                 + [random.random()]            # a bias weight for the bias input
                 for _ in range(num_neuron_in_a_hidden_layer)]    # for each and every hidden neurons

# Initialize weight vector for each neuron in the OUTPUT layer
# each output neuron has one weight per hidden neuron (in the last hidden layer), plus a bias weight
output_layer = [[random.random() 
                for _ in range(num_neuron_in_a_hidden_layer)]   # each output neuron has num_neuron_in_a_hidden_layer inputs
                + [random.random()]                             # a bias weight for the bias input
                for _ in range(output_size)]                    # for each and every of the output_size output neurons
# Initialize the network
network = [hidden_layer, output_layer]

# Iterate backpropagation many times so that hopefully weight vectors of each neuron 
# converting while satisfying the constrain of minimizing the error between the targets
# and network outputs (on training sets)
print("\nInitial network:")
print(*network, sep="\n")
for _ in range(10000):
    for training_input, target in zip (training_inputs, targets):
        backpropagate(network, training_input, target)
print("Learned network:")
print(*network, sep="\n")
print()
print("Its prediction performance:")
print("-------------ON TRAINING SETS:")
for i, training_input in enumerate(training_inputs):
    nn_prediction = predict_with_neural_network(network, training_input)
    print("Input = ")
    print(*[training_input[i:i+5] for i in range(0,25,5)], sep = "\n")
    print("Prediction from neural network:", *[round(x,2) for x in nn_prediction])
print("-------------ON TESTING SETS:")
for i, testing_input in enumerate(testing_inputs):
    nn_prediction = predict_with_neural_network(network, testing_input)
    print("Input = ")
    print(*[testing_input[i:i+5] for i in range(0,25,5)], sep = "\n")
    print("Prediction from neural network:", *[round(x,2) for x in nn_prediction])

plt.figure()
for i in range(5):  
    plt.subplot(2,3,i+1) 
    show_weights(network, 0, i)
    plt.title("neuron[0][{0:1d}] with bias = {1:.2f}".format(i, network[0][i][-1]))
plt.show()