'''
Created on Jul 23, 2017

@author: loanvo
'''
from collections import Counter
from c04_linear_algebra import distance_of_two_vecs
import random

def raw_majority_vote(labels):
    # counting the number of each label and return "winning-label" which has the biggest count
    # this function DOES NOT solve the "tie" situation in which more than one label have the same biggest counts
    label_counts = Counter(labels)
    winning_label = label_counts.most_common(1)[0]
    return winning_label

"""
For "tie" situation (has more than one winning_label), we have several solutions:
1. Pick one of the winners at random
2. Weight the votes by distance and pick the weighted winner.
3. Reduce k until we find a unique winner
"""

def majority_vote(labels): # assume that labels are sorted according to their distances
    # counting the number of each label and return "winning-label" which has the biggest count
    # this function SOLVE the "tie" situation
    label_counts = Counter(labels)
    winning_label, biggest_count = label_counts.most_common(1)[0]
    number_of_winning_labels = len([count for count in label_counts.values() if count==biggest_count])
    if number_of_winning_labels == 1: #if there is no more than one label has the biggest count
        return winning_label
    else:
        return majority_vote(labels[:-1]) # try again after removing one neighbor which is farthest - again assuming that labels are sorted by distances

def knn_classify(k, labeled_points, new_point):
    # each labeled_point should be a pair (point_locations, label)
    sorted_points = sorted(labeled_points, key=lambda labeled_point: distance_of_two_vecs(labeled_point[0], new_point)) 
    # labels of k nearest points
    k_nearest_labels = [label[-1] for label in sorted_points[:k]]
    return majority_vote(k_nearest_labels) # let them vote

# Generate random points
def random_point(dim):
    return [random.random() for _ in range(dim)]

# generate random distances (distance between random point pairs)
def random_distance(dim, num_pairs):
    return [distance_of_two_vecs(random_point(dim), random_point(dim)) for _ in range(num_pairs)]

