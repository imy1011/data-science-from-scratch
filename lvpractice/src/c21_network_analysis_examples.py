'''
Created on Aug 6, 2017

@author: loanvo
'''

from c21_network_analysis import add_user_info, shortest_paths_from, betweenness_centrality, \
closeness_centrality, make_adjacency_matrix, find_eigenvector, page_rank

"""
Betweenness Centrality
"""

users = [
    { "id": 0, "name": "Hero" },
    { "id": 1, "name": "Dunn" },
    { "id": 2, "name": "Sue" },
    { "id": 3, "name": "Chi" },
    { "id": 4, "name": "Thor" },
    { "id": 5, "name": "Clive" },
    { "id": 6, "name": "Hicks" },
    { "id": 7, "name": "Devin" },
    { "id": 8, "name": "Kate" },
    { "id": 9, "name": "Klein" }
]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

add_user_info(users, friendships)
#print(users)

print("\nShortest paths:")
for user in users:
    user["shortest_paths"] = shortest_paths_from(users, user["id"])
    print("from user", user["id"], "to")
    for to_user, shortest_paths_to_user in user["shortest_paths"].items():
        print(" user", to_user, ":", shortest_paths_to_user)
        
betweenness_centrality(users)
print("\nBetweeness centrality:")
for user in users:
    print(" user", user["id"], ": {0:.3f}".format(user["betweenness_centrality"]))

closeness_centrality(users)
print("\nCloseness centrality:")
for user in users:
    print(" user", user["id"], ": {0:.3f}".format(user["closeness_centrality"]))    
    
        
"""
EIGENVECTOR CENTRALITY
"""

adjacency_matrix = make_adjacency_matrix(users)
print(*adjacency_matrix, sep = "\n")
eig_vec, _ = find_eigenvector(adjacency_matrix)
print("\nEigenvector centrality:")
for eig_vec_i in eig_vec: print("{:.3f}".format(eig_vec_i))


"""
Directed Graphs and PageRank
"""
print("\n\nEndorsement:")
endorsements = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2),
                (2, 1), (1, 3), (2, 3), (3, 4), (5, 4),
                (5, 6), (7, 5), (6, 8), (8, 7), (8, 9)]
for user in users:
    user["endorses"] = []       # add one list to track outgoing endorsements
    user["endorsed_by"] = []    # and another to track endorsements
for source, target in endorsements:
    users[source]["endorses"].append(target)
    users[target]["endorsed_by"].append(source)
    
# find the most_endorsed
sorted_users = sorted(users, key = lambda user: len(user["endorsed_by"]), reverse = True)
for user in sorted_users:
    print(user["name"], "(id =", user["id"], ") endorsed by", len(user["endorsed_by"]), ":", user["endorsed_by"])

# page rank:
print("\n\nPage rank:")
page_rank(users, damping = 0.85, num_iters = 100)
for user in users:
    print(user["name"], "(id =", user["id"], ") has page-rank = {0:.3f}".format(user["page_rank"]))
