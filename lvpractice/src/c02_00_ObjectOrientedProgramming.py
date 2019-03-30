'''Imagine we didn't have the build-in Python set. 
Then we might to create our own Set class'''
class MySet():
    def __init__(self,values=None):
        '''This is the constructor. It gets called when you create a new Set'''
        self.dict = {}
        if values:
            for value in values:
                self.add(value)
    def __repr__(self):
        '''This is the string representation of a Set object'''
        return "MySet is {" + ', '.join(map(str,self.dict.keys())) + '}'
    def add(self,newItem):
        self.dict[newItem] = 1 #adding an item into the set is similar to adding a key into the its dict element and the value of the key is not important - here I assign value 1 to all key
    def remove(self,delItem):
        if delItem in self.dict:
            del self.dict[delItem]
        else:
            print('{} is NOT in the set. Do nothing!'.format(delItem))
    def contains(self,checkedItem):
        return checkedItem in self.dict    
s = MySet([0,2,3])
print(s)
s.add(4)
print('After adding 4,',s) 
s.add(1)
print('After adding 1, checking if 1 is in the set:',s.contains(1))  
s.remove(3)
print('After removing 3,',s)
print('Checking if 3 is in the set:',s.contains(3))
 
d = {'Tim':30,'Janet':20,'Jimmy':13,'Ann':8}
