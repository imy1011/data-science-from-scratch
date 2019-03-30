'''
Created on Aug 08, 2017

@author: loanvo
'''

from c04_linear_algebra import dot_product, scalar_multiply, vector_add
import math
from collections import defaultdict

"""
Recommending What’s Popular
"""

def most_popular_new_interests(user_interests, popular_interests, max_results=5):
    """
    Suggest to a user the most popular interests that he’s not already interested in:
    """
    suggestions = [interest for interest, _ in popular_interests if interest not in user_interests]
    return suggestions[:max_results]
        

"""
User-Based Collaborative Filtering
"""

def cosine_similarity(v, w):
    """
    It measures the “angle” between v and w. If v and w point in the same direction, 
    then the numerator and denominator are equal, and their cosine similarity equals 1. 
    If v and w point in opposite directions, then their cosine similarity equals -1.
    """
    return dot_product(v,w) / math.sqrt(dot_product(v, v)*dot_product(w, w))
        
def make_user_interest_vector(user_interests, unique_interests):
    """
    For each user, produce an “interest” row vector of 0s and 1s corresponding to the absence/existence of
    user's interests with the entries in the unique_interests vector
    """
    return [1 if interest in user_interests else 0 for interest in unique_interests]

def most_similar_users_to(user_id, user_interest_matrix):  
    """
    finds the most similar users to a given user
    """
    user_interest = user_interest_matrix[user_id]
    user_similarities = [cosine_similarity(user_interest, other_user_interest) for other_user_interest in user_interest_matrix]
    similarities = [(other_user_id, similarity)
                    for other_user_id, similarity in enumerate(user_similarities)
                    if other_user_id != user_id and similarity != 0]
    sorted_similarities = sorted(similarities, key = lambda similarity: similarity[1], reverse = True)
    return sorted_similarities

def user_based_suggestions(user_id, user_interest_matrix, unique_interests, include_current_interests = False):  
    """
    For each interest, we can just add up the user-similarities of the other users interested in it.
    Then we recommends the ones having the highest user-similarities
    """ 
    similarities = most_similar_users_to(user_id, user_interest_matrix)
    
    total_similarity_scores = [0 for _ in unique_interests]
    for other_user_id, similarity_score in similarities:
        total_similarity_scores = vector_add(total_similarity_scores, 
                                             scalar_multiply(similarity_score, 
                                                             user_interest_matrix[other_user_id]))
   
    
    interest_similar_score_pairs = [(interest_name, total_similarity_scores[interest_id])
                                    for interest_id, interest_name in enumerate(unique_interests)
                                    if include_current_interests or 
                                    user_interest_matrix[user_id][interest_id] != 1]
    
    interest_similar_score_pairs = sorted(interest_similar_score_pairs, key = lambda pair: pair[1], reverse = True)
    return interest_similar_score_pairs



    
"""
Item-Based Collaborative Filtering
"""    
def most_similar_interests_to(interest_id, interest_similarities, unique_interests):
    interest_similarWeights = [(unique_interests[other_interest_id], similarWeight) 
                               for other_interest_id, similarWeight in 
                               enumerate(interest_similarities[interest_id])
                               if similarWeight > 0 and other_interest_id != interest_id]
    return sorted(interest_similarWeights, key = lambda pair: pair[1], reverse = True)

def item_based_suggestions(user_id, users_interests, unique_interests, interest_similarities, include_current_interests=False):
    user_interest_names = users_interests[user_id]
    suggestions = defaultdict(float)
    for user_interest_name in user_interest_names:
        user_interest_id = unique_interests.index(user_interest_name)
        for interest_name, weight in  most_similar_interests_to(user_interest_id, interest_similarities, unique_interests):
            suggestions[interest_name] += weight
    suggestions = sorted([(interest_name, weight) for interest_name, weight in suggestions.items()],
                         key = lambda pair: pair[1], reverse = True)
    if not include_current_interests:
        suggestions = [(interest_name, weight) for interest_name, weight in suggestions 
                       if interest_name not in user_interest_names]
    return suggestions


    


