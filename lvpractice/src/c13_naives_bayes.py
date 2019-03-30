'''
Created on Jul 24, 2017

@author: loanvo
'''

import re
from collections import defaultdict
import math

"""
Note: Our codes for spam-email classifier with Naives Bayes are based on the following assumptions:
- P(word_i, word_j,... | spam) = P(word_i | spam)P(word_j | spam)...  (conditional independence assumption)
- P(spam) = P(non_spam) = 1/2 . Therefore: 
P(spam|words) = P(words|spam)p(spam) / [P(words|spam)p(spam) + P(words|non-spam)p(non-spam)]
              = P(words|spam) / [P(words|spam) + P(words|non-spam)
"""

def tokenize(message):
    """ tokenize messages into distinct words """
    #all_words = re.split("\s+", message.lower(), flags = re.IGNORECASE) # this pattern will return words including all non-alphabet words/characters:, !, ., 
    all_words = re.findall("[a-z0-9]+", message.lower())
    return set(all_words) #set: remove words that appear more than once

def  count_words(train_data):
    """ 
    Goal: Count the words in a labeled training set of messages.
    Input: Training set consists of pairs (message, is_spam)
    Output: return a dictionary whose keys are words whose values are two-element lists 
    [spam_count, non_spam_count] corresponding to how many times we saw that word in both 
    spam and nonspam messages
    """
    counts = defaultdict(lambda: [0, 0])
    for message, is_spam in train_data:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts

def word_probabilities(counts, total_spams, total_non_spams, k=0.5):
    """ Turn the word_counts into a dict: {w: [p(w | spam), p(w | ~spam)]}"""
    for word, (spam, non_spam) in counts.items():
        counts[word] = [(k+spam)/(2*k+total_spams), (k+non_spam)/(2*k+total_non_spams)]
    return counts 


def spam_probability(word_probs, message):
    """
    Use word probabilities (and our Naive Bayes assumptions) (learned from training sets)
    to assign probabilities to messages
    """
    """
    LOAN's NOTES: I don't think that it is correct to calculate 
    prob(all_words_in_training_messages|spam). I think we should calculate
    prob(all_words_in_testing_message|spam) = P(word_1_in_testing_message|spam)
    """
    prob_word_spam = 0
    prob_word_non_spam = 0
    for word in word_probs.keys():
        if word in tokenize(message):
            prob_word_spam += math.log(word_probs[word][0])
            prob_word_non_spam += math.log(word_probs[word][1])
        else:
            prob_word_spam += math.log(1-word_probs[word][0])
            prob_word_non_spam +=  math.log(1-word_probs[word][1])
    prob_all_words_given_spam = math.exp(prob_word_spam)
    prob_all_words_given_non_spam = math.exp(prob_word_non_spam)
    return prob_all_words_given_spam/(prob_all_words_given_spam+prob_all_words_given_non_spam)

class NaiveBayesClassifier:
    """
    Naive Bayes Classifier
    """
    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []
    def train(self, train_data):
        counts = count_words(train_data)
        total_spams = sum([1 if is_spam else 0 for _, is_spam in train_data])
        total_non_spams = len(train_data) - total_spams
        self.word_probs = word_probabilities(counts, total_spams, total_non_spams, self.k)
    def classify(self, message): 
        return spam_probability(self.word_probs, message)
    
def p_spam_given_word(word_prob):   
    """
    Input: (word_i, p(word_i | spam), p(word_i | non_spam))
    Output: p(spam | word_i)
    Again, assuming that p(spam) = p(non-spam) = .5
    """
    prob_word_given_spam, prob_word_given_non_spam = word_prob
    return prob_word_given_spam/(prob_word_given_spam+prob_word_given_non_spam)
         
    