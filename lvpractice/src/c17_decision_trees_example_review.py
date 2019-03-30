'''
Created on Oct 9, 2018

@author: loanvo
'''

from collections import defaultdict, Counter
import numpy as np

def get_attributes(inputs):
    all_attributes = set()
    for an_input in inputs:
        all_attributes.update(an_input[0].keys())
    return all_attributes

def partition_by(att, inputs):
    partitions = defaultdict(list)
    for an_input in inputs:
        attribute_label = an_input[0].get(att)
        partitions[attribute_label].append(an_input)
    return partitions

def get_branches(att, inputs):
    branches = defaultdict(list)
    for input_attributes, input_label in inputs:
        branch_att = input_attributes.get(att)
        if branch_att != None: #if input_attributes does not inlcude this attribute, there is no need to delte it
            del input_attributes[att]
        branches[branch_att].append((input_attributes, input_label))
    if None not in branches.keys():
        branches[None] = [({}, Counter([an_input[1] for an_input in inputs]).most_common(1)[0][0])]
    return branches

def partition_entropy_by(partitions):
    partition_entropy = 0
    partition_size = 0
    for subpartition in partitions.values():
        all_subpart_labels = [an_entry[1] for an_entry in subpartition]
        subpartition_counts = Counter(all_subpart_labels)
        subpartition_size = sum(subpartition_counts.values())
        partition_size += subpartition_size 
        subpartition_pi = [count_i/subpartition_size for count_i in subpartition_counts.values()]
        subpartition_entropy = sum([-p_i*np.log2(p_i) for p_i in subpartition_pi])
        partition_entropy += subpartition_entropy*subpartition_size
    return partition_entropy/partition_size

def build_tree_id3(inputs):
    all_attributes = get_attributes(inputs)
    all_labels = [an_input[1] for an_input in inputs]
    label_counts = Counter(all_labels)
    if len(label_counts) != 1 and len(all_attributes) != 0: #only one label --> tree ends, i.e, this is a leaf
        min_entropy = None
        for att in all_attributes:
            partition_entropy = partition_entropy_by(partition_by(att, inputs))
            if min_entropy==None or partition_entropy <= min_entropy:
                min_entropy = partition_entropy
                splitting_attribute = att
        branches = get_branches(splitting_attribute, inputs)
        tree = (splitting_attribute, {branch_att: build_tree_id3(branch_input) for branch_att, branch_input in branches.items()})            
    else:
        tree = label_counts.most_common(1)[0][0]  ######
    return tree

def print_tree(tree, space_string=''):
    vertical_line='  |___'
    no_vertical_line = ' ___'
    if isinstance(tree, tuple):
        print('\n', space_string, vertical_line, tree[0], end='')
        for k, v in tree[1].items():
            print('\n', space_string, '\t', vertical_line, k, end='')
            print_tree(v, space_string + '\t\t')
    else:
        print(no_vertical_line, tree, end='')
    if space_string=='':
        print('\n\n')

def decision_tree_classification(tree, test):
    flag = 1
    while flag:
        temp = tree[1].get(test.get(tree[0]))
        if temp != None:
            tree = temp
        else:
            tree = tree[1].get(None)
        if not isinstance(tree, tuple):
            flag = 0
            label = tree
    return label

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


all_attributes = get_attributes(inputs)
"""
for att in all_attributes:
    print("*******", att, "*******")
    partitions = partition_by(att, inputs)
    print(partitions)
    print("The corresponding entropy: {}".format(partition_entropy_by(partitions)))
"""
mytree = build_tree_id3(inputs)
print(mytree)
print_tree(mytree)
an_input = {'lang': 'Java', 'phd': 'no', 'level': 'Junior', 'tweets': 'yes'}
print(an_input, decision_tree_classification(mytree, an_input))
an_input = {'lang': 'Java', 'phd': 'yes', 'level': 'Junior', 'tweets': 'yes'}
print(an_input, decision_tree_classification(mytree, an_input))
an_input = {'level': 'Intern'}
print(an_input, decision_tree_classification(mytree, an_input))
an_input = {'tweets': 'not sure', 'level': 'Senior'}
decision_tree_classification(mytree, an_input)
print(an_input, decision_tree_classification(mytree, an_input))
an_input =  {'level': 'Senior'}
print(an_input, decision_tree_classification(mytree, an_input))
