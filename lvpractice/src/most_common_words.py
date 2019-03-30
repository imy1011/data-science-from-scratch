'''
Created on Jul 7, 2017
By Loan Vo
Original File Path: /Users/loanvo/GitHub/py/joelgrus/joelgrus/src/most_common_words.py

Descriptions:
- Goal: counting the words in its input and writes out the most common ones
- Input: 
    + stream from stdin
    + number of the most-common words
- Output:
    + most common words and their frequencies
'''

from collections import Counter
import sys

try:
    num_common_word = int(sys.argv[1]) #sys.argv is a list of string --> we have to convert into integer
except:
    num_common_word = 1
    print("WARNING: since you don't provide the number of most common words, only one will be printed out.")

read_lines = sys.stdin.read() # read_lines: a string containing everything from stdin including \n
word_counts = Counter(read_lines.lower().split())
for most_common_word, word_frequency in word_counts.most_common(num_common_word):
    #print(word_frequency,"\t", most_common_word) # we want to practice stdout, so we are not using print here
    sys.stdout.write(str(word_frequency)) # have to convert the integer number word_frequency into str as stdout only handles string
    sys.stdout.write("\t")
    sys.stdout.write(most_common_word)
    sys.stdout.write("\n")

