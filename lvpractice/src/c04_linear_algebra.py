'''
Created on July 2, 2017

@author: loanvo
'''

# lists/lists of lists are not convenient in handling vector/matrix. In this chapter, the author only use list to represent vector or matrix
# Loan will do what the author do but also use numpy to refresh the material in her head



import math

##
def magnitude(w):
    return math.sqrt(sum_of_squares(w))

##
def vector_add(v,w):
    return [ v[i] + w[i] for i, _ in enumerate(v) ]


##
def vector_subtract(v,w):
    return [ v[i] - w[i] for i, _ in enumerate(v) ]

##
def distance_of_two_vecs(v,w):
    return magnitude(vector_subtract(v, w))

##
def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot_product(v, v)

##
def squared_distance(v,w):
    return sum_of_squares(vector_subtract(v, w))

##
'''
def vector_sum(*list_vectors):
    return reduce(vector_add,list_vectors) 
'''
def vector_sum(*list_vectors):
    sum_of_all_vectors = list_vectors[0]
    for v_i in list_vectors[1:]:
        sum_of_all_vectors = vector_add(sum_of_all_vectors, v_i)
    return sum_of_all_vectors

##
def scalar_multiply(c,v):
    return [c*vi for vi in v]

##
def vector_mean(*vectors):
    #print(vector_sum(*vectors))
    return scalar_multiply(1/len(vectors),vector_sum(*vectors))

##
def dot_product(v,w):
    return sum([vi*wi for vi, wi in zip(v, w)])

##
def shape(A):
    try:
        num_cols = len(A[0]) #if A has only one dim, len(A[0]) will give error
    except:
        num_cols = 1
    num_rows = len(A)    
    return (num_rows, num_cols)

def get_row(A,i):
    return A[i]

def get_column(A,j):
    return [A_i[j] for A_i in A]

def make_matrix(num_rows, num_cols, entry_fn):
    return [[entry_fn(i,j) for j in range(num_cols)] for i in range(num_rows)]

def is_diagonal(i,j):
    return 1 if j==i else 0

def matrix_transpose(A):
    return [ [Ai[j] for Ai in A] for j in range(len(A[0])) ]

def augmented_matrix(x,y):
    '''
    num_rows, num_cols = shape(x)
    def add_a_column(i, j):
        return x[i][j] if j < num_cols else y[i]
    data = make_matrix(num_rows, num_cols + 1, add_a_column)
    '''
    # Simpler method:
    data = [x[i] + [y[i]] for i in range(len(y))]
    return data

def matrix_multiply(A, B):
    """ return matrix production of A*B """
    rA, cA = shape(A)
    rB, cB = shape(B)
    if cA != rB:
        raise "Number of columns of matrix A must be equal to number of of rows of matrix B"
    else:
        return make_matrix(rA, cB, lambda i, j: dot_product(get_row(A, i), get_column(B, j)))
    
def vector_as_matrix(v):
    """ transform n-dim row vector (1xn matrix) into n-dim column vector or nx1 matrix"""
    return [[v_i] for v_i in v]
    
def vector_from_matrix(v_as_matrix):
    """returns the n x 1 matrix as a list of values"""
    """i.e. transform n-dim column vector or nx1 matrix into n-dim row vector (1xn matrix)"""
    return [v_i[0] for v_i in v_as_matrix]

def matrix_operate(A, v):
    """
    Input: - v: 1xn matrix (row vector)
           - A: mxn matrix
    Output: (A*v')'
    """
    return vector_from_matrix(matrix_multiply(A, vector_as_matrix(v)))

