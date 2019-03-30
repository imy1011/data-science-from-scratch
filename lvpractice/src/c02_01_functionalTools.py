'''
This is about: partial, map, filter and reduce
List comprehensive: map, filter and reduce
'''
from _functools import reduce

'''parial'''
from functools import partial
def exp(base,pow,addingAmount=0):
    return base**pow+addingAmount
def two_to_the(pow):
    return exp(2,pow)
print(two_to_the(3))

#Using partial to define a new function with partial application
#of the given argument
#here we provide in advance the 1st argument of the 
#function exp, i.e., base = 2 in the exp function
anotherTwoToThe = partial(exp,2) 
print(anotherTwoToThe(3))

#we also can specify the argument name and the other arguments
# will be provided later according to its input order 
somethingToThe2 = partial(exp,pow=3)
print(somethingToThe2(addingAmount=3,base=4))
#should be the same as
somethingToThe2 = partial(exp,pow=3,base=4)
print(somethingToThe2(addingAmount=3))

'''map'''
def double(x):
    return x*2

xs = [3, 2, -1, 5]
double_xs = [double(x) for x in xs]
print(double_xs)
anotherDouble_xs = list(map(double,xs))
print(anotherDouble_xs)
list_double_func = partial(map,double)
print(list(list_double_func(xs)))

#Map can also be used with multiple-argurment functions if you provide multiple lists:
def multiply(x,y):return x*y
multiplyOfTwoLists = list(map(multiply,[1,2,0],[3,4,8]))
print(multiplyOfTwoLists)


'''
filter
'''
def is_even(x): return x%2==0
print([x for x in xs if is_even(x)])
print()
print(list(filter(is_even,xs)))
print()
list_evener = partial(filter,is_even)
print(list(list_evener(xs)))
'''
reduce
'''
print('---------------')
import _functools
x_product = _functools.reduce(multiply,xs)
print(x_product)
list_product = partial(_functools.reduce,multiply)
print(list_product(xs))

import matplotlib as mpl
from matplotlib import pyplot as plt
plt.xti