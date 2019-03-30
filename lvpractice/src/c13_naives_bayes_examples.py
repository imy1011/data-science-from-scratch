'''
Created on Jul 24, 2017

@author: loanvo
'''
from c11_machine_learning import split_data
import re
import glob
import random
from c13_naives_bayes import NaiveBayesClassifier, p_spam_given_word
from collections import Counter

"""
Three folders: spam, easy_ham, and hard_ham. Each folder contains many emails, 
each contained in a single file. 
To keep things really simple, we’ll just look at the subject lines of each email
"""


# get all training messages (only their subject line) and put them into "data"
path_to_emails = "/Users/loanvo/datascience/joelgrus/data/emails/*/*"
data = []
for fn in glob.iglob(path_to_emails):
    is_spam = 'ham' not in fn
    with open(fn,'r', encoding='latin-1') as email_file_handle:
        email_content_as_a_string = ''.join(email_file_handle.readlines())
        #can not use subject_line = re.findall(...)[0] as re.findall might return an empty list)
        subject_line = str(re.findall("^Subject:.*$", email_content_as_a_string, flags = re.MULTILINE))
        data.append((re.sub("Subject:\s*","",subject_line),is_spam))
        

# split the data into training data and test data
random.seed(0)
train_data, test_data = split_data(data,.75)

# let's initialize and train our classifier
classifier = NaiveBayesClassifier()
classifier.train(train_data)

# let test our spam classifier. Return: # triplets (message, actual is_spam label, predicted spam probability)
classified = [(message_content, true_label, classifier.classify(message_content))
              for message_content, true_label in test_data]

# Assume that if predicted spam_probability > 0.5, message is predicted as spam
# Let's count the number of correct prediction:
counts = Counter([(true_label, spam_prob>0.5) for _, true_label, spam_prob in classified])
tp, fp, fn, tn = [counts[(true_label, predicted_label)] 
                  for true_label in [True, False] 
                  for predicted_label in [True, False] ]

# let's sort the testing message by its predict spam_probability
classified.sort(key= lambda triplets : triplets[2])
# the 5 messages that have the highest spam_probability while they are actually non-spam ones
five_highest_predicted_spam_prob_among_the_non_spam = [triplets for triplets in classified if not triplets[1]][-5:]
print("Non-spam messages with highest predicted spam_prob:")
print(*five_highest_predicted_spam_prob_among_the_non_spam, sep="\n")
# the 5 messages that have the lowest spam_probability while they are actually spam ones
five_lowest_predicted_spam_prob_among_the_spam = [triplets for triplets in classified if triplets[1]][:5]
print("\nSpam messages with lowest predicted spam_prob:")
print(*five_lowest_predicted_spam_prob_among_the_spam, sep="\n")

# words that appears in spam messages with highest probabilities
words_sorted_by_spam_prob_given_word = sorted(classifier.word_probs, key = lambda x: p_spam_given_word(classifier.word_probs[x]))
word_that_makes_the_message_most_likely_to_be_spam = words_sorted_by_spam_prob_given_word[-5:]
word_that_makes_the_message_less_likely_to_be_spam = words_sorted_by_spam_prob_given_word[:5]
print("\nWords that make a message containing its most likely to be a spam")
print(*word_that_makes_the_message_most_likely_to_be_spam,sep="\n")
print("\nWords that make a message containing its less likely to be a spam")
print(*word_that_makes_the_message_less_likely_to_be_spam,sep="\n")
