'''
Created on Jul 30, 2017

@author: loanvo
'''

import random
from c04_linear_algebra import vector_mean,squared_distance,\
    distance_of_two_vecs
from matplotlib import image as mpimage
from blaze import inf


class KMeans:
    """performs k-means clustering"""
    def __init__(self, k):
        self.k = k          # number of clusters
        self.means = None   # means of clusters. Its initial value is None
    def classify(self, invec):
        """return the index of the cluster closest to the input"""
        return min(range(self.k), key = lambda i: squared_distance(self.means[i], invec))
    def train(self, invecs):
        # choose k random points as the initial means
        self.means = random.sample(invecs, self.k)
        assignments = None
        # 
        while 1:
            # Find new assignments
            new_assignments = list(map(self.classify, invecs)) #if we don't convert into list here, the loop "for i in range(self.k)" having access to assignments/new_assignments only run successfully for the first round, as map is iterator and its next method would point to empty after the 1st round
            # If no assignments have changed, we're done.
            if assignments == new_assignments:
                return assignments
            else: 
                # Otherwise keep the new assignments
                assignments = new_assignments
                # And compute new means based on the new assignments
                for i in range(self.k):
                    # find all the points assigned to cluster i
                    invecs_in_the_same_cluster = [invect for invect, assignment in 
                                                  zip(invecs, assignments) 
                                                  if assignment == i]
                    if len(invecs_in_the_same_cluster): # if has more than one element, calculate their means
                        self.means[i] = vector_mean(*invecs_in_the_same_cluster) 
                              
        
def squared_clustering_errors(invecs, k):
    """finds the total squared error from k-means clustering the invecs"""
    clusterer = KMeans(k)
    assignments = clusterer.train(invecs)
    error_vecs = [squared_distance(invec, clusterer.means[assignment]) 
                  for invec, assignment in zip(invecs, assignments)]
    return sum(error_vecs)
    
def recolor(impath = r"/Users/loanvo/Downloads/fall2015_ss.jpg", num_colors = 5): 
    imvecs = mpimage.imread(impath)
    r, c, _ = imvecs.shape
    pixvecs = [list(map(float,pix)) for row in imvecs for pix in row]
    clusterer = KMeans(num_colors)
    assignments = clusterer.train(pixvecs)
    print(clusterer.means)
    cc_pixvecs = [clusterer.means[assignment] for assignment in assignments]
    cc_imvecs = [[cc_pixvecs[row*c + col] for col in range(c)] for row in range(r)]
    return imvecs, cc_imvecs
   
    
def is_leaf(cluster):
    """a cluster is a leaf if it has length 1"""
    return len(cluster) == 1
    
def get_children(cluster):
    """returns the two children of this cluster if it's a merged cluster;
    raises an exception if this is a leaf cluster"""
    if is_leaf(cluster):
        raise TypeError("a leaf cluster has no children")
    else:
        return cluster[1]
    
def get_values(cluster):
    """returns the value in this cluster (if it's a leaf cluster)
    or all the values in the leaf clusters below it (if it's not)"""
    if is_leaf(cluster):
        return cluster # be careful here! cluster is already a 1-tuple containing value
                        # Should not return cluster[0]. 
                        # Because although cluster[0] is only a single vector, it will be considered
                        # as an iterable --> the loop "for value in get_values(child)" would
                        # consider each vector element as a vector
    else:
        return [value for child in get_children(cluster) 
                for value in get_values(child)]
    
    
def cluster_distance(cluster1, cluster2, distance_agg=min):
    """compute all the pairwise distances between cluster1 and cluster2
    and apply _distance_agg_ to the resulting list"""
    return distance_agg([distance_of_two_vecs(value1, value2) 
                         for value1 in get_values(cluster1) 
                         for value2 in get_values(cluster2)])
       
def get_merge_order(cluster):
    if is_leaf(cluster):
        return inf
    else:
        return cluster[0]

def bottom_up_cluster(invecs, distance_agg=min):
    # start with every input a leaf cluster / 1-tuple
    clusters = [(invec,) for invec in invecs]
    while len(clusters)>1:
        paired_clusters = [(cluster1, cluster2) 
                           for i, cluster1 in enumerate(clusters) 
                           for cluster2 in clusters[i+1:]]
        c1, c2 = min(paired_clusters, 
                     key = lambda c12: cluster_distance(c12[0], c12[1], distance_agg))
        clusters = [c for c in clusters if c!=c1 and c!=c2]
        merged_cluster = (len(clusters),[c1, c2])
        clusters.append(merged_cluster)
            
    return clusters[0]

def base_cluster_string(base_cluster, num_tab = 1):
    if len(base_cluster) == 2 :
        s = "(" + str(base_cluster[0]) + ", [" + \
        base_cluster_string(base_cluster[1][0], num_tab = num_tab+1) + \
        ", \n" + " " *5* num_tab + \
        base_cluster_string(base_cluster[1][1], num_tab = num_tab+1) + \
        "]" + ")"
    else:
        s = "(" + str(base_cluster[0]) + ",)"
    return s    
def generate_clusters(base_cluster, num_clusters):
    # start with a list with just the base cluster
    clusters = [base_cluster]
    # as long as we don't have enough clusters yet...
    while len(clusters)<num_clusters:
        # choose the last-merged of our clusters
        next_cluster = min(clusters, key = get_merge_order)
        clusters = [cluster for cluster in clusters if cluster != next_cluster]
        clusters.extend(get_children(next_cluster))  #Note: clusters.extend() is different from clusters.append()
                                                    # get_children returns a list (of 2 elements). 
                                                    # If using clusters.append(), it will add the list (one entry) into clusters
                                                    # if using clusters.extend(), it will add each and every element (len(list) entries) of the list into clusters
    return clusters
