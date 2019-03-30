'''
Created on Jul 7, 2017
By Loan Vo
Original File Path: /Users/loanvo/GitHub/py/joelgrus/joelgrus/src/line_count.py

Descriptions:
- Goal: Count the number of lines from an input file object 
- Input: 
    + a file object from stdin
- Output
    + number of lines
'''
import sys

line_count = 0
for _ in sys.stdin:
    line_count += 1
print("Total number of lines:", line_count)
