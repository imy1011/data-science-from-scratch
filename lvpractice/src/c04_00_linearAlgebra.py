'''
Writing the functions to add/multiply vector
These actually are available for use in numpy package
'''
import numpy as np
l1 = [1,5, -2, 6]
l2 = [0, 3, 1, -3]
a = 2
v1 = np.array(l1)
v2 = np.array(l2)
def add_vectors(l1,l2):
    if len(l1)==len(l2):
        l=[l1i+l2i for l1i, l2i in zip(l1,l2)]
    else:
        l=None   
    return l

def scalar_multiplication(l1,a):
    l=[l1i*a for l1i in l1]
    return l
print('Adding vector by using function for list:',add_vectors(l1,l2))
print("Adding vector by using numpy lib:",v1+v2)
print('Multiplied by scalar by using function for list:',scalar_multiplication(l1, a))
print("Multiplied by scalar with numpy lib:",v1*a)
