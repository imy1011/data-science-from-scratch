'''
Created on Aug 3, 2017

@author: loanvo
'''

from collections import defaultdict, Counter
import random
import requests
from bs4 import BeautifulSoup
import re


"""
WORD-CLOUD
"""
def text_size(total):
    """equals 8 if total is 0, 28 if total is 200"""
    return 8 + total/200*20 
    
"""
N-GRAM
"""
def fix_unicode(text):
    """
    The apostrophes in the text are actually the Unicode character u"\u2019". 
    We’ll create a helper function to replace them with normal apostrophes
    """
    return text.replace(u"\u2019", "'")

def get_document(url):
    # use requests to get a website at http://radar.oreilly.com/2010/06/what-is-data-science.html
    html = requests.get(url)
    # use BeautifulSoup to parse the html document. Find article-body div 
    parsed_html = BeautifulSoup(html.text, 'html5lib')
    content = parsed_html.find('div', 'article-body')
    # split the content word by word (including the period ".") and put each and every of them into a list 
    # (a sequence of words)
    document = []
    for paragraph in content("p"):
        # matches a word or a period (i.e. sentences)
        words = re.findall(r"[\w']+|[\.]", fix_unicode(paragraph.text)) # Find all words (including word having "'" such as we've) or period
        document.extend(words)
    return document   

def make_bigram_transitions(doc_data):
    bigram_transitions = defaultdict(list)
    for prev_word, curr_word in zip(doc_data[:], doc_data[1:]):
        bigram_transitions[prev_word].append(curr_word)
    return bigram_transitions
    
def generate_using_bigrams(bigram_transitions):
    curr_word = "."
    result = []
    while True:
        next_words = bigram_transitions[curr_word]
        curr_word = random.choice(next_words)
        result.append(curr_word)
        if curr_word == ".": return " ".join(result)
    
def make_trigram_transitions(doc_data):
    trigram_transitions = defaultdict(list)
    start_words = [] # List of words that begin a sentence
    for prev_word, curr_word, next_word in zip(doc_data, doc_data[1:], doc_data[2:]):
        trigram_transitions[(prev_word, curr_word)].append(next_word)
        if prev_word == ".": 
            start_words.append(curr_word)
    return trigram_transitions, start_words
    
def generate_using_trigrams(trigram_transitions, start_words):
    prev_word = "."
    curr_word = random.choice(start_words)
    result = [curr_word]
    while True:
        next_words = trigram_transitions[(prev_word, curr_word)]
        prev_word = curr_word
        curr_word = random.choice(next_words)
        result.append(curr_word)
        if curr_word == ".": return " ".join(result)
        
"""
GRAMMARS: given some sets of grammar rules, construct a machinenary generates sentences based on those rules
"""
def is_terminal(token):
    return token[0][0] != "_"
    
def expand(grammar, tokens):
    results = []
    for token in tokens:
        choice_for_a_token = random.choice(grammar[token]).split()
        if is_terminal(choice_for_a_token):
            result =  choice_for_a_token
        else:
            result = expand(grammar, choice_for_a_token)
        results.extend(result)
    return results # should not return a string here with the syntax: 
                # results if len(results)==1 else " ".join(results) 
                # as if inner iteration return a string then 
                # "result" is a string --> results.extend would extend 
                #with each and every single character of a string
            
def generate_sentence(grammar):
    return " ".join(expand(grammar, ["_S"]))


"""
GIBBS SAMPLING
Gibbs sampling is a technique for generating samples from multidimensional distributions 
when we only know some of the conditional distributions.
The way Gibbs sampling works is that we start with any (valid) value for x1,..., x_(i-1), x_i, x_(i+1),..., xn
and then repeatedly alternate replacing xi with a random value picked conditional on x1,..., x_(i-1), x_(i+1),..., xn 
and replacing y with a random value picked conditional on x. 
After a number of iterations, the resulting values of x1,..., x_(i-1), x_i, x_(i+1),..., xn will represent a sample 
from the UNCONDITIONAL JOINT distribution
"""
def roll_a_dice():
    """return a result when randomly roll a dice"""
    return random.randint(1,6)
    
def direct_sample():
    """return (x,y) where x is the 1st dice rolling result and 
    y is the summation of rolling the 1st and 2nd dices"""
    x = roll_a_dice()
    y = x + roll_a_dice()
    return (x, y)
    
def random_y_given_x(x):
    """return y = the summation of rolling the 1st and 2nd dices 
    given the result of rolling the 1st dices"""
    return roll_a_dice() + x
    
def random_x_given_y(y):
    """return x = the result of rolling the 1st dices
    given y = the summation of rolling the 1st and 2nd dices """
    if y<=7:
        x = random.randint(1,y-1)
    else:
        x = random.randint(y-6, 6)
    return x
    
def gibbs_sample(num_iters=100):
    x = random.randint(1, 6)
    for _ in range(num_iters):
        y = random_y_given_x(x)
        x = random_x_given_y(y)
    return (x, y)    
    
    
def compare_distributions(num_samples=1000):
    """ COMPARING the results of sampling directly (x,y) and gibbs sampling (x,y)"""
    counts = defaultdict(lambda: [0, 0])
    for _ in range(num_samples):
        counts[direct_sample()][0] += 1
        counts[gibbs_sample(100)][1] += 1
    return counts
        
        
"""
TOPIC MODELING
"""
def sample_from(weights):
    """returns i with probability weights[i] / sum(weights)"""
    total_weights = sum(weights)
    random_number = total_weights*random.random()# a uniform distributed random number ranging [0, total_weights]
    for i, weight in enumerate(weights):
        random_number -= weight         # return the smallest i 
        if random_number <= 0: return i # such that weights[0] + ... + weights[i] >= rnd
        
def topic_modeling(documents, total_num_of_topics = 4):
    def p_topic_given_document(topic, d, alpha=0.1):
        """the fraction of words in document _d_
        that are assigned to _topic_ (plus some smoothing)"""
        return (topic_counts_in_a_doc[d][topic] + alpha)/ \
            (total_words_in_each_doc[d] + total_num_of_topics*alpha)
       
    def p_word_given_topic(word, topic, beta=0.1):
        """the fraction of words assigned to _topic_
        that equal _word_ (plus some smoothing)"""
        return (word_counts_for_each_topic[topic][word] + beta)/ \
            (topic_counts_in_all_docs[topic] + num_of_distinct_words_in_all_docs*beta)
    
    def topic_weight(d, word, k):
        """given a document and a word in that document,
        return the weight for the kth topic"""
        return p_word_given_topic(word, k) * \
            p_topic_given_document(k, d)
    
    def choose_new_topic(d, word):
        weights = [topic_weight(d, word, k) for k in range(total_num_of_topics)]
        return sample_from(weights)
    
    # documents: list of documents. each element of a list is a document which includes a list of words
   
    # count the number of distinct words in all documents:
    num_of_distinct_words_in_all_docs = len(set([word for doc in documents for word in doc]))
    
    # Initialize topics: randomly assign a topic to each word in the document
    random.seed(0)
    doc_topics = [[random.randrange(total_num_of_topics) for _ in doc] for doc in documents]
    
    # Count the number of words in each document:
    total_words_in_each_doc = [len(doc) for doc in documents]
                
    # Count the total number of each and every topic appearing in each document
    topic_counts_in_a_doc = [Counter(topics_in_each_doc) for topics_in_each_doc in doc_topics]
    # Count the total number of each and every topic appearing in ALL documents
    topic_counts_in_all_docs = Counter([topic for topics_in_each_doc in doc_topics
                                        for topic in topics_in_each_doc])
    # counting the number of times a certain word is assigned to the same topic (on over all docs)
    word_counts_for_each_topic = [Counter() for _ in range(total_num_of_topics)]
    for d in range(len(documents)):
        for word, topic in zip(documents[d], doc_topics[d]):
            word_counts_for_each_topic[topic][word] += 1 
    """        
    # My variable names are different from the one in the book, and here I just want to make sure that 
    # although their names are different but they have the same roles.
    document_lengths = list(map(len, documents))
    document_topic_counts = [Counter() for _ in documents]
    topic_word_counts = [Counter() for _ in range(total_num_of_topics)]
    topic_counts = [0 for _ in range(total_num_of_topics)]
    for d in range(len(documents)):
        for word, topic in zip(documents[d], doc_topics[d]):
            document_topic_counts[d][topic] += 1
            topic_word_counts[topic][word] += 1
            topic_counts[topic] += 1
            
    print("------------------------")
    print(document_lengths)
    print(total_words_in_each_doc)
    print(document_lengths==total_words_in_each_doc)
    print(document_topic_counts)
    print(topic_counts_in_a_doc)
    print(document_topic_counts==topic_counts_in_a_doc)
    print(topic_word_counts)
    print(word_counts_for_each_topic)
    print(topic_word_counts==word_counts_for_each_topic)
    print(topic_counts)
    print(topic_counts_in_all_docs)
    print("------------------------")
    """
    for _ in range(1000):    
        # choose new topic for each word based on topic weight in document d
        for d in range(len(documents)):
            for t, (word, current_topic) in enumerate(zip(documents[d], doc_topics[d])):
                # remove this word / topic from the counts
                # so that it doesn't influence the weights
                topic_counts_in_a_doc[d][current_topic] -= 1
                topic_counts_in_all_docs[current_topic] -= 1
                word_counts_for_each_topic[current_topic][word] -=1
                total_words_in_each_doc[d] -= 1
                # choose a new topic based on the weights
                new_topic = choose_new_topic(d, word)
                # update topic of the current word
                doc_topics[d][t] = new_topic
                # and now add it back to the counts
                topic_counts_in_a_doc[d][new_topic] += 1
                topic_counts_in_all_docs[new_topic] += 1
                word_counts_for_each_topic[new_topic][word] +=1
                total_words_in_each_doc[d] += 1
            
            
    return topic_counts_in_a_doc, word_counts_for_each_topic        
                
            
    