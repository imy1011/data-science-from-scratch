'''
Created on Jul 26, 2017

@author: loanvo
'''


from c17_decision_trees import partition_entropy_by, partition_by, build_tree_id3, classify


"""
Preparing input data
"""
inputs = [
    ({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'no'},    False),
    ({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'yes'},   False),
    ({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'no'},      True),
    ({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'no'},   True),
    ({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'no'},       True),
    ({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'yes'},     False),
    ({'level':'Mid', 'lang':'R', 'tweets':'yes', 'phd':'yes'},         True),
    ({'level':'Senior', 'lang':'Python', 'tweets':'no', 'phd':'no'},  False),
    ({'level':'Senior', 'lang':'R', 'tweets':'yes', 'phd':'no'},       True),
    ({'level':'Junior', 'lang':'Python', 'tweets':'yes', 'phd':'no'},  True),
    ({'level':'Senior', 'lang':'Python', 'tweets':'yes', 'phd':'yes'}, True),
    ({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'yes'},     True),
    ({'level':'Mid', 'lang':'Java', 'tweets':'yes', 'phd':'no'},       True),
    ({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, False)
]


"""
Step by step to build a decision tree
"""

for key in ['level','lang','tweets','phd']:
    print(key, partition_entropy_by(inputs, key))
print("\n\n")
# the lowest entropy comes from attribute "level". 
# "level" has 3 possible values: "Junior", "Mid", "Senior"
# We need to divide our inputs into 3 corresponding subsets
# Subset of "level = Mid" has all labels = True --> leaf node (no subtree)
# Subset of "level = senior" has different label values --> partitioning this subset
senior_inputs = partition_by(inputs, "level")["Senior"]
print("Senior inputs:", senior_inputs)
for key in ['lang','tweets','phd']:
    print(key, partition_entropy_by(senior_inputs, key))
print("\n\n")
# Subset of "level = Junior" has different label values --> partitioning this subset
junior_inputs = partition_by(inputs, "level")["Junior"]
print("Junior inputs:", junior_inputs)
for key in ['lang','tweets','phd']:
    print(key, partition_entropy_by(junior_inputs, key))
print("\n\n")

"""
Building the whole tree
"""
tree = build_tree_id3(inputs)
print("Our tree:", tree[0])
print(*tree[1].items(), sep = "\n")

"""
Test the decision
"""
print("\n\n")
print("Classification results based on the above decision tree:")
#
an_input = {"level" : "Junior", "lang" : "Java", "tweets" : "yes", "phd" : "no"} #True
print("----------\nInput: ", an_input)
print(classify(tree, an_input))
#
an_input = {"level" : "Junior", "lang" : "Java", "tweets" : "yes", "phd" : "yes"} #False
print("----------\nInput: ", an_input)
print(classify(tree, an_input))
#
an_input = { "level" : "Intern" }
print("----------\nInput: ", an_input)
print(classify(tree, an_input)) # True
#
an_input = { "level" : "Senior", "tweets" : "not sure"}
print("----------\nInput: ", an_input)
print(classify(tree, an_input)) # False
#
an_input = { "level" : "Senior" }
print("----------\nInput: ", an_input)
print(classify(tree, an_input)) # False