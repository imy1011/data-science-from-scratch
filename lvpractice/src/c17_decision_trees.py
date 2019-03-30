'''
Created on Jul 26, 2017

@author: loanvo
'''

import math
from collections import Counter, defaultdict
from _functools import partial

def entropy(class_probs):
    """given a list of class probabilities, compute the entropy"""
    return sum([-pi*math.log2(pi) for pi in class_probs if pi != 0])
    
def class_probabilities(labels):
    """Input: list of labels
    Output: class probabilities """
    total_count = len(labels)
    return [each_label_count/total_count for each_label_count in Counter(labels).values()]
    
def data_entropy(labeled_data):   
    """Input: a data set: list of pairs (attribute_dict, label)
    Output: entropy of the set"""
    labels = [label for _, label in labeled_data]
    return entropy(class_probabilities(labels))

def partition_entropy(subsets):
    """
    if we partition our data S into subsets S1, S2, ... Sm containing portion q1, q2, ..., qm of the 
    data, then we compute the entropy of the partition as a weighted sum:
    H = q1 * H(S1) + q2 * H(S2) + ... + qm * H(Sm)
    Find the entropy from this partition of data into subsets
    Subsets is a list of lists of labeled data
    """
    total_count = sum([len(subset) for subset in subsets])
    return sum([len(subset) / total_count * data_entropy(subset)
                for subset in subsets])
  
def partition_by(inputs, attribute):
    """each input is a pair (attribute_dict, label).
    returns a dict : attribute_value -> inputs"""
    subsets = defaultdict(list)
    for an_input in inputs:
        subsets[an_input[0][attribute]].append(an_input)
    return subsets
    
def partition_entropy_by(inputs, attribute):
    """computes the entropy corresponding to the given partition""" 
    return partition_entropy(list(partition_by(inputs, attribute).values()))

def build_tree_id3(inputs, split_candidates=None):
    """Building a tree with "greedy" algorithm ID3"""
    # if this is the very first iteration of build_tree_id3, we haven't had split_candidates yet
    # So if split_candidates = None, we construct split_candidates
    if split_candidates is None: #None type is different from an empty list, i.e, type([]) is list
        split_candidates = list(inputs[0][0].keys())
        
    # If all labels of inputs are either all True or all False, then return a leaf Nodes
    num_of_trues = sum([1 if label == True else 0 for _, label in inputs])
    num_of_falses = len(inputs) - num_of_trues
    if num_of_falses == 0: #all Trues
        return True
    elif num_of_trues == 0: # all False
        return False
    # If not, we need to continue to split the tree based on the split_candidates
    
    # Check if split_candidates is empty
    if len(split_candidates) == 0:
        return num_of_trues > num_of_falses   # return True if the majority of the labels are true. \
                                            # Otherwise return False
        
    # Calculate entropy of the partition corresponding to each split_candidate
    # and choose the one with lowest entropy to be the decision node
    split_attribute = min(split_candidates, key = partial(partition_entropy_by,inputs))
    new_split_candidates = [split_candidate 
                            for split_candidate in split_candidates 
                            if split_candidate != split_attribute]
    subsets = partition_by(inputs, split_attribute)
    
    #recursively build subtrees
    subtrees = {subset_attribute: build_tree_id3(subset, new_split_candidates) 
                for subset_attribute, subset in subsets.items()}
    # If an input_data needed to be classified by the decision tree has any missing attribute 
    # its classification result will take the majority label at the latest decision node
    subtrees[None] = num_of_trues > num_of_falses
    
    return (split_attribute, subtrees)
            
    
    
def classify(tree, input_data):
    """classify the input using the given decision tree"""
    #print(tree)
    
    # if we reach the leaf node, stop the iteration and return the label
    if tree is False or tree is True:
        return tree
    
    split_attribute = tree[0]
    subtree_key = input_data.get(split_attribute) # if input_data doesn't have the 
                                                    # split_attribute key (missing value), 
                                                    # then we assign subtree_key = None
    subtree_dict = tree[1]
    if subtree_key not in subtree_dict.keys(): # input_data has an unexpected value
        subtree_key = None
    subtree = subtree_dict[subtree_key]    
    #print("****", subtree)
    
    return classify(subtree, input_data)
    
def forest_classify(trees, input_data):
    """
    Given how closely decision trees can fit themselves to their training data, it’s not surprising 
    that they have a tendency to overfit. One way of avoiding this is a technique called 
    random forests, in which we build MULTIPLE decision trees and let them vote on how to classify 
    inputs
    """
    labels = [classify(tree, input_data) for tree in trees]
    return Counter(labels).most_common(1)[0][0]
    
    
    
  
    
  
