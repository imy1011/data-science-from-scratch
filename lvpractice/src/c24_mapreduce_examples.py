'''
Created on Aug 08, 2017

@author: loanvo
'''

from c24_mapreduce import word_count_old, word_count, map_reduce, wc_mapper, wc_reducer,\
    sum_reducer, data_science_day_mapper, words_per_user_mapper,\
    most_popular_word_reducer, count_distinct_reducer, liker_mapper, convert_sparse_matrix_to_entries,\
    matrix_multiply_mapper, matrix_multiply_reducer,\
    convert_entries_to_a_sparse_matrix
import datetime
from c04_linear_algebra import shape
from functools import partial

"""
Word-counting without and with map-reduce and map-reduce framework
"""
documents = ["data science", "big data", "science fiction"] # 3 documents 
print("Counting number of each word in all documents by NOT using Map-reduce:")
print(word_count_old(documents))
print("\nCounting number of each word in all documents by using Map-reduce:")
print(word_count(documents))
print("\nCounting number of each word in all documents by using Map-reduce general framework:")
print(map_reduce(documents, wc_mapper, wc_reducer))

"""
Example: Analyzing Status Updates
"""
print("\n\nExample: Analyzing Status Updates")
status_updates = [{"id": 1,
                   "username" : "joelgrus",
                   "text" : "Is anyone interested in a data science book?",
                   "created_at" : datetime.datetime(2013, 12, 21, 11, 47, 0),
                   "liked_by" : ["data_guy", "data_gal", "mike"]},
                  {"id": 2,
                   "username" : "marysmith",
                   "text" : "Dinner is ready. My data scientist guys aren't home yet.",
                   "created_at" : datetime.datetime(2016, 1, 3, 1, 7, 20),
                   "liked_by" : ["cookiegirl", "naughtydaughter", "mike"]},
                  {"id": 1,
                   "username" : "joelgrus",
                   "text" : "Book! Have this year election data.",
                   "created_at" : datetime.datetime(2016, 11, 18, 9, 52, 30),
                   "liked_by" : ["data_guy", "data_gal"]},
                  {"id": 3,
                   "username" : "data_gal",
                   "text" : "Just cook the whole Data Science from Scratch book written by joel grus",
                   "created_at" : datetime.datetime(2017, 3, 18, 20, 26, 10),
                   "liked_by" : ["data_guy", "joelgrus"]},
                  {"id": 4,
                   "username" : "mike",
                   "text" : "Data science is progressing fast",
                   "created_at" : datetime.datetime(2012, 2, 7, 3, 7, 10),
                   "liked_by" : ["joelgrus"]},
                  {"id": 2,
                   "username" : "marysmith",
                   "text" : "How many holes does a strainer have? Too many.",
                   "created_at" : datetime.datetime(2009, 10, 13, 12, 6, 32),
                   "liked_by" : []}
                  ]

data_science_days = map_reduce(status_updates, data_science_day_mapper, sum_reducer)
print("The number of time the word 'data science' appearing in blog statuses on each weekday:")
print(data_science_days)

print("\nThe most common word that each user puts in her status updates")
print(*map_reduce(status_updates, words_per_user_mapper, most_popular_word_reducer), sep = "\n")

print("\nThe number of distinct status-liker for each user:")
print(*map_reduce(status_updates, liker_mapper, count_distinct_reducer), sep = "\n")

#words_and_counts = words_per_user_mapper(status_updates[5])
#most_popular_word_reducer(user_id, words_and_counts)
"""
Example: Matrix Multiplication
"""
A = [[3, 2, 0],
     [0, 0, 0]]

B = [[4, -1, 0],
     [10, 0, 0],
     [0, 0, 0]]

num_of_rows = shape(A)[0]
num_of_cols = shape(B)[1]
entries = convert_sparse_matrix_to_entries(A = A, B = B)
print("Non-zero entries of sparse matrixes A and B:", entries)
AB_product = map_reduce(entries, partial(matrix_multiply_mapper, num_of_rows, num_of_cols), matrix_multiply_reducer)
print("Matrix product of A and B is matrix C = ")
print(*convert_entries_to_a_sparse_matrix(AB_product, num_of_rows, num_of_cols), sep = "\n")