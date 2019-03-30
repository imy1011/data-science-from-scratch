'''
Created on Jul 19, 2017
By Loan Vo
Original File Path: /Users/loanvo/GitHub/py/joelgrus/joelgrus/src/c04_linear_algebra_examples.py

Descriptions:
- Goal:
- Input:
- Output:
'''

import numpy as np
from c04_linear_algebra import vector_add, vector_subtract, vector_sum, \
    scalar_multiply, vector_mean, dot_product, make_matrix, is_diagonal

# lists/lists of lists are not convenient in handling vector/matrix. In this chapter, the author only use list to represent vector or matrix
# Loan will do what the author do but also use numpy to refresh the material in her head

'''===========================================
VECTOR
'''

list_vector_1 = [1,2,5,9]
list_vector_2 = [3,-1,0,2]
list_vector_3 = [-3, 7, 1, -2]

#np_vector_0 = np.array([[1,2,5,9]])
np_vector_1 = np.array([1,2,5,9])
np_vector_2 = np.array(list_vector_2) #equivalent to: np_vector_2 = 3,-1,0,2]
np_vector_3 = np.array(list_vector_3)

#print(np_vector_0.shape)
#print(np_vector_1.shape)
#print("Vector 0:", np_vector_0)

##
print("Vector 1:", np_vector_1)
print("Vector 2:", np_vector_2)
print("Vector 3:", np_vector_3)


##
print("\nAdding vectors 1 and 2")
print("List approach:", vector_add(list_vector_1, list_vector_2))
print("numpy vector approach:", np_vector_1 + np_vector_2)


##
print("\nSubtracting vectors  2 from 1")
print("List approach:", vector_subtract(list_vector_1, list_vector_2))
print("numpy vector approach:", np_vector_1 - np_vector_2)


##
print("\nSumming vectors 1+2+3")
print("List approach:", vector_sum(list_vector_1, list_vector_2, list_vector_3))
print("numpy vector approach:", np.sum(np.stack((np_vector_1, np_vector_2, np_vector_3),axis = 0),axis = 0))


##
print("\nScalar multiply: 3*vector1")
print("List approach:", scalar_multiply(.5, list_vector_1))
print("numpy vector approach:", .5*np_vector_1)


##

print("\nVector mean")   
print("List approach:", vector_mean(list_vector_1, list_vector_2, list_vector_3))
print("numpy vector approach:", np.mean(np.stack((np_vector_1, np_vector_2, np_vector_3),axis = 0), axis = 0))


##
print("\nDot product")
print("List approach:", dot_product(list_vector_1, list_vector_2))
print("numpy vector approach:", np.dot(np_vector_1, np_vector_2))

##
identity_matrix = make_matrix(5,5,is_diagonal)
print(identity_matrix)
