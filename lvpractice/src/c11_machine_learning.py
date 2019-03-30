'''
Created on Jul 7, 2017
By Loan Vo
Original File Path: /Users/loanvo/GitHub/py/joelgrus/joelgrus/src/c11_machine_learning.py
'''
from c04_linear_algebra import augmented_matrix

"""
1. OVERFITTING vs. UNDERFITTING
    1.a. OVERFITTING: model performs well on training data sets but generalize poorly for any new data.
    That could involve:
    - learning noise in the data
    - identifying specific inputs rather than the factors that actually predicts for the desired outputs
    1.b. UNDERFITTING: model doesn't perform well even on training data
    
    Even when model performs well on TESTing data sets, there are some potential issues:
    - if many users appear in both the training and testing sets, the model might actually only
    identify users rather than discover relationship involving attributes
    - if training/testing split is not only used to judge a model but also to choose among different
    models --> although each model might not be overfit, the fact that "choosing a model that performs
    best on the test set" makes the test set actually a 2nd TRAINING set --> Should split data into
    3 parts: 
        + training set for building models
        + a validation set for choosing among trained models
        + a test set for judging the final model
    
    
2. CORRECTNESS
    Given the true and predicted labels, there are four classification results:
    - True positive: tp
    - False positive: fp
    - True negative: tn
    - False negative: fn
    
    2.a. ACCURACY = (tp + tn)/(tp+fp+tn+fn)
    NOT really meaningful especially in cases where true positive >> (or <<) true negative 
    
    2.b. PRECISION AND RECALL
    https://en.wikipedia.org/wiki/Precision_and_recall
    
    - Precision = tp/(tp+fp): a measure of exactness or quality
        + In pattern recognition, information retrieval, binary classification, precision ~ 
        the fraction of relevant instances among the retrieved instances
        + In search engine, precision ~ how useful the search results are
    
    - Recall = tp/(tp+fn): a measure of completeness or quantity
        + In pattern recognition, information retrieval, binary classification, recall ~ 
        the fraction of relevant instances that have been retrieved over total relevant 
        instances in the image
        + In search engine, recall ~ how complete the results are
        
    Example: brain surgery to remove cancer tumor.
        
    --> F1 score (harmnonic mean): combining of precision and recall: 2*p*r/(p+r).
    And it can be proved that: min(p,r) <= F1 <= 2*min(p,r)
    
    
3. THE BIAS-VARIANCE TRADE-OFF
    3.a. BIAS ~ learning or prediction error: high bias = big error on training data sets (model 
    performs poorly even on training sets)
    
    3.b. VARIANCE ~ how much learned model changes when using different training sets.
    
    --> high bias + low variance ~ underfitting
    --> low bias + high variance ~ overfitting
    
    SOLUTIONS:
    - High bias: 
        + probably need to ADD MORE FEATURES (more training data won't help)
    - High variance:
        + REMOVE some FEATURES
        + ADD MORE TRAINING data
    
    
4. FEATURE EXTRACTION AND SELECTION
Three feature types:
    - Binary: yes/no or 1/0
    - number
    - discrete set of options
Depend on the feature types, we have constrains the type of models we can use
    - Naive Bayes classifier: suited for yes-no features
    - Regression model: requires numeric features (can include dummy variables that are 0s and 1s)
    - Decision trees: can deal with numeric or categorical data   
How we choose features: combination of experience and domain expertise
"""
import random

def split_data(data, prob):
    """
    split data into fractions [prob, 1-prob]
    """
    train = []
    test = []
    for datum in data:
        if random.random()<prob:
            train.append(datum)
        else:
            test.append(datum)
    return train, test

def train_test_split(x, y = None, test_pct = .33):
    if y is None:
        data = x # in case x and y is already combined
    else:
        data = augmented_matrix(x, y)
    train, test = split_data(data, 1 - test_pct)
    
    x_train = [train_i[:-1] for train_i in train] # magical un-zip trick doesn't work in python 3
    y_train = [train_i[-1] for train_i in train] # magical un-zip trick doesn't work in python 3
    x_test = [test_i[:-1] for test_i in test] 
    y_test = [test_i[-1] for test_i in test] 
    return x_train, x_test, y_train, y_test 

def accuracy(tp, fp, fn, tn):
    return (tp+tn)/(tp+fp+fn+tn)

def precision(tp, fp):
    return tp/(tp+fp)

def recall(tp,fn):
    return tp/(tp+fn)

def f1_score(tp,fn,fp):
    precision_score = precision(tp,fp)
    recall_score = recall(tp,fn)
    return 2*precision_score*recall_score/(precision_score+recall_score)