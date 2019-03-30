'''
Created on Aug 6, 2017

@author: loanvo
'''


from collections import deque
from c04_linear_algebra import make_matrix, shape, magnitude, matrix_operate, scalar_multiply,\
    distance_of_two_vecs
import random

"""
Betweenness Centrality
"""

def add_user_info(users, friendships):
    for user in users:
        user["friends"] = list()            # give each user a friends list
    for id1, id2 in friendships:
        users[id1]["friends"].append(id2)   # add friends' ids into friend list
        users[id2]["friends"].append(id1)

def shortest_paths_from(users, from_user):
    shortest_paths_to = {from_user: [[]]}
    frontier = deque([(from_user, friend) for friend in users[from_user]["friends"]])
    while len(frontier):
        pre_user, user = frontier.popleft()
        # Paths from "from_user" to "user" are established 
        # by adding/extending paths "from_user"-"pre_user" and "pre_user"-"user".
        # And since "user" is neighbor of "pre_user", the extending portion is simply [user]
        new_paths_to = [shortest_paths_to_pre_user + [user]
                        for shortest_paths_to_pre_user in shortest_paths_to[pre_user]]
        # there might exist shortest paths to "user"
        old_paths_to = shortest_paths_to.get(user, [])  # if there are shortest_paths_to "user", we place 
                                                        # them into old_paths_to, if not old_paths_to = []
        len_of_old_paths = float('inf') if len(old_paths_to) == 0 else len(old_paths_to[0])
        # compare the length of new_paths_to with old_paths_to and only add new_paths_to to the list of 
        # shortest_paths_to if len(new_paths_to) is equal (or smaller) than old_paths_to.
        if len(new_paths_to[0]) < len_of_old_paths:
            shortest_paths_to[user] = new_paths_to
        elif len(new_paths_to[0]) == len_of_old_paths:
            for np in new_paths_to:
                if np not in old_paths_to:
                    shortest_paths_to[user].append(np)
        # add friends of "user" to the queue frontier as there are paths from from_user to user's friends
        # via from_user - user - user's friends
        # However, we won't add to the queue frontier if shortest_paths_to has already had the shortest
        # paths from from_user - user's friend . It is because the existing path  from_user - user's friend
        # certainly shorter than any paths we are going to find by the queue pair (user, user's friend)
        # In other words, we would add the new pair (user, user's friend) into the queue froniter if and only
        # if user's friend is NOT a key in the dictionary shortest_paths_to
        for friend_to_user in users[user]["friends"]:
            if friend_to_user not in shortest_paths_to.keys() and (user, friend_to_user) not in frontier:
                frontier.append((user, friend_to_user)) 
  
    return shortest_paths_to

def betweenness_centrality(users):
    for user in users:
        user["betweenness_centrality"] = 0

    num_of_users = len(users)
    for from_id in range(num_of_users):
        for to_id in range(from_id+1,num_of_users):
            shortest_paths = users[from_id]["shortest_paths"][to_id]
            num_of_shortest_paths = len(shortest_paths)
            for shortest_path in shortest_paths:
                for id_on_shortest_paths in shortest_path:
                    if id_on_shortest_paths != to_id:
                        users[id_on_shortest_paths]["betweenness_centrality"] += 1/num_of_shortest_paths

def closeness_centrality(users):
    def farness(user):
        """the sum of the lengths of all the user's shortest paths to other users"""
        return sum([len(paths[0]) for _, paths in user["shortest_paths"].items()])
    for user in users:
        user["closeness_centrality"] = 1/farness(user)
        
        
"""
EIGENVECTOR CENTRALITY
"""
def make_adjacency_matrix(users):
    nr = len(users)
    nc = len(users)
    adjacency_matrix = make_matrix(nr, nc, lambda i, j: 1 if j in users[i]["friends"] else 0)
    return adjacency_matrix

def find_eigenvector(A, tolerance=0.00001):
    _, nc = shape(A)
    eig_vec = [random.random() for _ in range(nc)]
    eig_vec = scalar_multiply(1/magnitude(eig_vec), eig_vec)
    while True:
        next_eig_vec = matrix_operate(A, eig_vec)
        eig_val = magnitude(next_eig_vec)
        next_eig_vec = scalar_multiply(1/eig_val, next_eig_vec)
        if distance_of_two_vecs(eig_vec, next_eig_vec) <= tolerance:
            return eig_vec, eig_val
        eig_vec = next_eig_vec
        

"""
Directed Graphs and PageRank
"""
"""
A simplified version looks like this:
- There is a total of 1.0 (or 100%) PageRank in the network.
- Initially this PageRank is equally distributed among nodes.
- At each step, a large fraction of each node’s PageRank is distributed evenly among its outgoing links.
- At each step, the remainder of each node’s PageRank is distributed evenly among all nodes.
More reference: https://en.wikipedia.org/wiki/PageRank
"""
def page_rank(users, damping = 0.85, num_iters = 100):
    num_of_users = len(users)
    pr = {user["id"]: 1 / num_of_users for user in users}   # Initially a total pagerank of 1 is equally \
                                                            # distributed among nodes.
    base_pr = (1-damping)/num_of_users  # small fraction of each node's page rank is distributed evenly
                                        # among all nodes --> each node receives
                                        # pr_i * (1-damping)/num_of_users) from node i --> each node receives
                                        # (1-damping)/num_of_users from all node
    for _ in range(num_iters):
        next_pr = {user["id"]: base_pr for user in users} # the small fraction that each node receives from all nodes
        for user in users:
            user_pr = pr[user["id"]] * damping
            endorses = user["endorses"]
            # if a user doesn't endorse any other users or a page doesn't have any out-going link,
            # its rank will be divided equally for all other pages
            # Joel Grus's book fails taking care of this situation
            if len(endorses) == 0:
                endorses = [endorse["id"] for endorse in users if endorse["id"] != user["id"]]
            for endorse in endorses:
                next_pr[endorse] += user_pr/len(endorses)   # the large fraction that a node gives to 
                                                            # each of its endorses 
        pr = next_pr
        #print(sum(pr.values())) #should be equal to 1 in each iteration
    
    for user in users:
        user["page_rank"] = pr[user["id"]]        
                    
                
