'''
Created on AUg 08, 2017

@author: loanvo
'''
from collections import Counter
from c22_recommender_systems import most_popular_new_interests,\
    make_user_interest_vector, most_similar_users_to, user_based_suggestions,\
    cosine_similarity, most_similar_interests_to, item_based_suggestions
from functools import partial
from c04_linear_algebra import matrix_transpose



users_interests = [
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

"""
Recommending What’s Popular
"""

# For each interest, count the number of users who are interested in it.
print("Popular interests:")
popular_interests = Counter([interest for user_interests in users_interests 
                             for interest in user_interests]).most_common()
print(*popular_interests, sep = "\n")
# suggesting new interest based on the most popular interests
user_id = 1
print("\nUser's current interest:", users_interests[user_id])
suggestions_with_popular_interest = most_popular_new_interests(users_interests[user_id], popular_interests)
print("Suggested  new interests (based on popular interests):")
print(suggestions_with_popular_interest)


"""
User-Based Collaborative Filtering
One way of taking a user’s interests into account is to look for users who are somehow similar to him, 
and then suggest the things that those users are interested in.
Limitation:
This approach doesn’t work as well when the number of items gets very large. Recall the curse of 
dimensionality from Chapter 12—in large-dimensional vector spaces most vectors are very far apart
(and therefore point in very different directions). That is, when there are a large number of interests 
the “most similar users” to a given user might not be similar at all.
"""

# using a set comprehension to find the unique interests, putting them in a list, and then sorting them
# Loan's note: don't think that sorting is necessary here.
unique_interests = sorted(list({interest for user_interest in users_interests for interest in user_interest}))
print("\nSorted unique interests from all users:")
print(unique_interests)
# create a matrix of user interests simply by map-ping this function against the list of lists of interests
# user_interest_matrix[i][j] equals 1 if user i specified interest j, 0 otherwise
user_interest_matrix = list(map(partial(make_user_interest_vector, unique_interests = unique_interests), \
                                users_interests))
print("\nUser interest matrix:")
print(*user_interest_matrix, sep = "\n")

user_id = 0
most_similar_users = most_similar_users_to(user_id, user_interest_matrix)
print("\nUsers having most similar interest with user_id = ", user_id)
print(*most_similar_users, sep = "\n")
print("\nand so his recommended interests (based on his similarity with other users' interests):")
print(*user_based_suggestions(user_id, user_interest_matrix, unique_interests), sep = "\n")


"""
Item-Based Collaborative Filtering
"""
interest_user_matrix = matrix_transpose(user_interest_matrix)
interest_similarities = [[cosine_similarity(interest_user_i, interest_user_j) 
                          for interest_user_i in interest_user_matrix]
                         for interest_user_j in interest_user_matrix]
print("\n\n")
interest_id = 0
print("Most similar interests to interest", interest_id, ":")
print(*most_similar_interests_to(interest_id, interest_similarities, unique_interests), sep = "\n")
user_id = 0
print("\nRecommendation for user", user_id)
print(*item_based_suggestions(user_id, users_interests, unique_interests, interest_similarities), sep = "\n")
