'''
Created on Aug 08, 2017

@author: loanvo
'''
from c13_naives_bayes import tokenize
from collections import Counter, defaultdict
from functools import partial
import re

"""
Word-counting without and with map-reduce and map-reduce framework
"""
def word_count_old(documents):
    """word count not using MapReduce"""
    return Counter([word for document in documents for word in tokenize(document)])
        
    
def wc_mapper(document):
    """for each word in the document, emit (word,1)"""
    for word in tokenize(document):
        yield (word,1)
    
def wc_reducer(word, counts):
    """sum up the counts for a word"""
    yield (word, sum(counts))

def word_count(documents):
    """count the words in the input documents using MapReduce"""
    collector = defaultdict(list)
    for document in documents:
        for word, count in wc_mapper(document):
            collector[word].append(count)
    return [output 
            for word, count_list in collector.items() 
            for output in wc_reducer(word, count_list)]

# General framework for map reduce
def map_reduce(inputs, mapper, reducer):
    """runs MapReduce on the inputs using mapper and reducer"""
    collector = defaultdict(list)
    for an_input in inputs:
        for key, value in mapper(an_input):
            collector[key].append(value)
    return [output 
            for key, values_in_list in collector.items() 
            for output in reducer(key, values_in_list)]
    
def reduce_values_using(aggregation_fn, key, values):
    """reduces a key-values pair by applying aggregation_fn to the values"""
    return (key, aggregation_fn(values))

def values_reducer(aggregation_fn):
    """turns a function (values -> output) into a reducer
    that maps (key, values) -> (key, output)"""
    return partial(reduce_values_using, aggregation_fn)

# Define some commonly-used values_reducer:
sum_reducer = values_reducer(sum)
max_reducer = values_reducer(max)
min_reducer = values_reducer(min)
count_distinct_reducer = values_reducer(lambda values_in_list: len(set(values_in_list)))

"""
Example: Analyzing Status Updates with map-reduce
"""
def data_science_day_mapper(status_update):
    """yields (day_of_week, 1) if status_update contains "data science" """
    if "data science" in status_update["text"].lower():
        yield (status_update["created_at"].weekday(), 1)

#find out for each user the mo st common word that she puts in her status updates
def words_per_user_mapper(status_update):
    for word in re.findall("[a-z0-9']+", status_update["text"].lower()):
        yield (status_update["username"], (word, 1))
        
def most_popular_word_reducer(user, words_and_counts):
    """given a sequence of (word, count) pairs,
    return the word with the highest total count"""        
    yield (user, Counter([word for word, _ in words_and_counts]).most_common(1))
    
def liker_mapper(status_update):
    """ find out the number of distinct status-likers for each user"""
    for liker in status_update["liked_by"]:
        yield (status_update["username"], liker)
    
"""
Example: Matrix Multiplication: for hug and sparse matrix multiplication C = A*B
C_i_k = sum(A_i_j * B_j_k)  
C_h_j = sum(A_h_i * B_i_j)
--> given a non-zero element of matrix A or B, i.e. A_i_j or B_i_j, we need to output:
- key:  the index of matrix C element, i.e. C_i_k or C_h_j
- value: (index in the summation j, A_i_j) or (index in the summation i, B_i_j)
"""
def matrix_multiply_mapper(A_rows, B_cols, element):
    """
    Input: 
    - A_rows: number of row in matrix A
    - B_cols: number of columns in matrix B
    - element: a tuple (matrix_name, i, j, value)
    Output:
    - a tuple: (C_row_index, C_column_index), (summation_index, value)
    """
    matrix_name, i, j, value = element
    if matrix_name == "A":
        for k in range(B_cols):
            yield ((i,k), (j, value))
    else:
        for h in range(A_rows):
            yield ((h, j), (i, value))
     

def matrix_multiply_reducer(key, indexed_values):
    ab_pairs = defaultdict(list)
    for idx, value in indexed_values:
        ab_pairs[idx].append(value) 
    yield (key, sum([ab_pair[0]*ab_pair[1] for ab_pair in ab_pairs.values() if len(ab_pair)==2]))
    
def convert_sparse_matrix_to_entries(**matrixes):
    """
    Input: dictionary {matrix_1_name : matrix_1_in_full_form, matrix_2_name : matrix_2_in_full_form,...}
    Output: [(matrix_1_name, row_index, column_index, non-zero-value-element), ...] 
    """
    return [(matrix_name, row_idx, col_idx, val) 
            for matrix_name, matrix in matrixes.items() 
            for row_idx, row in enumerate(matrix) 
            for col_idx, val in enumerate(row) if val != 0]

def convert_entries_to_a_sparse_matrix(entries, num_of_rows, num_of_cols):
    """
    Input: 
    """
    matrix_in_full_form = [[0 for _ in range(num_of_cols)] for __ in range(num_of_rows)] 
    for (row_idx, col_idx), value in entries:
        matrix_in_full_form[row_idx][col_idx] = value
    return matrix_in_full_form
        