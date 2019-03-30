'''
Created on Jun 23, 2017

@author: loanvo
'''
from statsmodels.graphics.tukeyplot import results
from numba.tests.test_random import np_extract_randomness


'''
List comprehensions
'''
even_numbers = [x for x in range(5) if x%2==0]; print(even_numbers)
square_dict = {x:x**2 for x in range(4)}; print(square_dict)
pairs = [(x,y) for x in range(10) for y in range(4)]; print(pairs)
keyValuePairs = {x:y for x in range(10) for y in range(4)}; print(keyValuePairs) #ATTENTION!!!!!!! key x are the same for many values y --> only one (key,value) remains for each x
increasingPairs = [(x,y) for x in range(10) for y in range(x+1,10)]; print(increasingPairs)

'''
Randomness
'''
import random
#random.seed: set an internal state based on which random module actually produces pseudo random number
#random.random : generate a random number in [0,1]
random.seed(10);print("Random seed was reset to 10:",random.random())
random.seed(5);print("Random seed was reset to 5:",random.random())
random.seed(10);print("Random seed was reset to 10:",random.random())
#randomly pick an element in a range
print("Randomly choose an element in a range [0,9]:", random.randrange(10))
print("Randomly choose an element in a range [3,9]: ", random.randrange(3,10))
#randomly shuffle elements
upToTen = list(range(10))
random.shuffle(upToTen) # in-place operation
print("Randomly shuffle elements from 0 to 10:", upToTen)
#randomly choose a sample of elements with/without replacement
print("Using random.choice to choose sample WITH replacement:",[random.choice(range(10)) for _ in range(6)]) #at each loop random.choice choose ONE sample in the list [0,9]
print("Using random.sample to choose sample WITHOUT replacement:",random.sample(range(10),6))



'''
Objected oriented programming: Class in python
'''
class MySet:
    def __init__(self,values):
        self.dict = {}
        if values is not None:
            for value in values:
                self.add(value)
    
    def add(self,value):
        self.dict[value] = True
        
    def contains(self,value):
        return value in self.dict
    
    def remove(self, value):
        self.dict.__delitem__(value)
        
    def allValues(self):
        return self.dict.keys()
        
s = MySet([1,2,3])
print("My set:",s.allValues())
s.add(4)
print(s.contains(4))
s.remove(3)
print(s.contains(3))


'''
Functional tools: functools.partial, map, reduce, filter
'''
#Functools
from functools import partial
def exp(base,power):
    return base**power
two_to_the = partial(exp,2) #input argument: function name + some of its arguments
print("Using functools.partial(funcName,someOfItsFirstInputArgument):",two_to_the(5))
square_of = partial(exp,power=3)
print("Using functools.partial(funcName,namedArgument=):", square_of(2))
#map
two_to_the_0_to4 = list(map(two_to_the,range(5)))
print("map(func,iterator): compute the function each value of iterator:",two_to_the_0_to4)
def multiply(x,y): 
    return x*y
print("map(func,iterator1,iterator2): compute the function each group values of iterators:", 
      list(map(multiply,[2, 4],[1, 7, 8])))
#filter
def multiple_of_three(x): return x%3==0
print("filter(function Or None, iterator): return values of iterator corresponding to which function returns true:", 
      list(filter(multiple_of_three,range(20))))
#reduce
from functools import reduce
print("reduce(func,list): applying func into the first two elements of the list,",
      "then continue applying func onto the previous output and the 3rd element of the list, and so on:",
      reduce(multiply,range(1,6)))


'''
enumerate
'''
my_text = "Wellness issue. Yoga."
for i, my_character in enumerate(my_text):
    print("Character ",i,": ", my_character)


'''
zip and argument unpacking
'''
list1=['a','b','c','d']
list2=[1,2,3]
pairs = list(zip(list1, list2))
print("zip: pairing/grouping element of each list together:", pairs)
print("unzip: unpacking a list using *arg:", list(zip(*pairs)))



'''
Iterables vs Iterators vs Generators
http://nvie.com/posts/iterators-vs-generators/
'''
x = [ 'a', 'b', 'c', 'd', 'e', 'f']
y = [ 0, 1, 2, 3, 4, 5 ]


xy = zip(x,y) # generator also is an iterator, i.e., it has a __next()__ method
i = 0
while i<3:
    print("--------i = ", i)
    # xy is a zip object and so after the first round (i=0), __next()__ raises a StopIteration exception
    # for later while-loop (i=1, 2, ...), the for-loop doesn't give any result as a result of xy's StopIteration exception 
    for xi, yi in xy: 
        print(xi, "and", yi)
    i +=1
    
j = 0
while j<3:
    print("zip_direct---j = ", j)
    for xi, yi in zip(x,y): # doesn't matter how many while-loop we have as we don't use xy here but zip(x,y)
        print(xi, "and", yi)
    j +=1
 
another_xy = zip(x,y)   
def test_zip(zipObject):
    for xi, yi in zipObject:
        print(xi, "and", yi)
k = 0
while k<3:
    print("zip-input-of-a-function---k = ", k)
    test_zip(another_xy)
    k +=1
    
    
'''
Python: pass by object reference
https://stackoverflow.com/questions/13299427/python-functions-call-by-reference
There are essentially three kinds of 'function calls':
- Pass by value
- Pass by reference
- Pass by object reference
Python is a PASS-BY-OBJECT-REFERENCE programming language. 
Also, called "passed by assignment":
https://stackoverflow.com/questions/986006/how-do-i-pass-a-variable-by-reference
'''
    


'''
*arg and **karg
'''
def magic(*arg, **karg):
    print("Unnamed arguments:",arg) # arg: tuple of all unnamed-inputs
    print("Unpacking unnamed args:", *arg) # *arg: unpacking tuple into multiple individual
    print("Named arguments:",karg)
    print("Unpacking named arguments' keys:",*karg)
    
magic(3,5,7,a=2,b=1)
