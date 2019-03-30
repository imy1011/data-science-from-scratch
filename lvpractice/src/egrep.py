'''
Created on Jul 7, 2017
By Loan Vo
Original File Path: /Users/loanvo/GitHub/py/joelgrus/joelgrus/src/egrep.py

Descriptions:
- Goal: scan each line of input file for a given pattern and output them. Ignore the unmatched ones.
- Input: 
    + a file
    + pattern to search for 
- Output
    + lines in the file that has the pattern 
'''

import sys, re

# sys.argv is a list containing command-line arguments passed to the python script http://www.pythonforbeginners.com/system/python-sys-argv
pattern_in_line = sys.argv[1] 

for read_line in sys.stdin: # sys.stdin is just another file object, which happens to be opened by Python before your program starts. What you do with that file object is up to you, but it is not really any different to any other file object, its just that you don't need an "open"
    if re.search(pattern_in_line, read_line): # if read_line doesn't have the pattern_in_line --> return None
        sys.stdout.write(read_line)  # write string to stream





