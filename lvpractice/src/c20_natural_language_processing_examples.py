'''
Created on Aug 3, 2017

@author: loanvo
'''

from matplotlib import pyplot as plt
from c20_natural_language_processing import text_size, get_document, make_bigram_transitions, \
generate_using_bigrams, make_trigram_transitions, generate_using_trigrams, generate_sentence, \
compare_distributions, topic_modeling
from collections import defaultdict
from rope.contrib.codeassist import get_doc
"""
WORD CLOUDS
"""
# Collection of data science–related buzzwords:
# Each word is accompanied by two numbers between 0 and 100 representing how frequently it appears 
# in job postings and on resumes
"""
data = [ ("big data", 100, 15), ("Hadoop", 95, 25), ("Python", 75, 50),
         ("R", 50, 40), ("machine learning", 80, 20), ("statistics", 20, 60),
         ("data science", 60, 70), ("analytics", 90, 3),
         ("team player", 85, 85), ("dynamic", 2, 90), ("synergies", 70, 0),
         ("actionable insights", 40, 30), ("think out of the box", 45, 10),
         ("self-starter", 30, 50), ("customer focus", 65, 15),
         ("thought leadership", 35, 35)]
# Approach 1: The word cloud approach is just to arrange the words on a page in a cool-looking font 
# Approach 2: scatter them so that x and y positions are corresponding to their popularity in job ad and resume 
plt.figure()
for the_text, job_pop, resume_pop in data:
    plt.text(job_pop, resume_pop, the_text, ha='center', va='center', 
             size=text_size(job_pop + resume_pop) )
plt.xlabel("Popularity on Job Postings")
plt.ylabel("Popularity on Resumes")
plt.axis([0, 100, 0, 100])
plt.xticks([])
plt.yticks([])
plt.show()
"""


"""
N-GRAM MODELS
"""

"""
# Getting a document and make them into a sequence of words (i.e. a list)
url = "http://radar.oreilly.com/2010/06/what-is-data-science.html"
document = get_document(url)

# construct the bigrams from the sequence of words
bigram_transitions = make_bigram_transitions(document)
# generate a sentence from bigram_transitions
bigram_sentence = generate_using_bigrams(bigram_transitions)
print("Example of bigram construction:", bigram_sentence)

# construct the trigrams from the sequence of words
trigram_transitions, start_words = make_trigram_transitions(document)
# generate a sentence from bigram_transitions
trigram_sentence = generate_using_trigrams(trigram_transitions, start_words)
print("Example of trigram construction:", trigram_sentence)
"""


"""
GRAMMARS
"""

grammar = {
        "_S"  : ["_NP _VP"],
        "_NP" : ["_N",
                 "_A _NP _P _A _N"],
        "_VP" : ["_V",
                 "_V _NP"],
        "_N"  : ["data science", "Python", "regression"],
        "_A"  : ["big", "linear", "logistic"],
        "_P"  : ["about", "near"],
        "_V"  : ["learns", "trains", "tests", "is"]
    }
print("\n\nGenerate sentences with n-grams:")
print(generate_sentence(grammar))


"""
GIBBS SAMPLING
"""
print("\n\nUsing gibbs sampling method to generate random samples of joint distributions \
    when only some conditional probabilities avaialbe:")
print(*compare_distributions(10000).items(), sep = "\n")


"""
TOPIC MODELING
"""
print("Topic Modeling: given a list of documents, find some most likely topics for each document:")
documents = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]

topic_counts_in_a_doc, word_counts_for_each_topic = topic_modeling(documents, total_num_of_topics = 4)
print("Most common words per topic:")
for i in range(4):
    print("\t - Topic", i, ":", word_counts_for_each_topic[i].most_common(5))
topic_names = ["Big Data and Programming Language",
               "Database",
               "Machine learning",
               "Statistics"]
for i, doc in enumerate(documents):
    print("\n")
    print(doc)
    for topic, count in topic_counts_in_a_doc[i].most_common(3):
        print(topic_names[topic], ":", count)
