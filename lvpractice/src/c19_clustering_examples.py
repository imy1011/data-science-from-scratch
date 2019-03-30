'''
Created on Jul 30, 2017

@author: loanvo
'''
from c19_clustering import KMeans, squared_clustering_errors, recolor, bottom_up_cluster, \
base_cluster_string, generate_clusters, get_values
import random
from matplotlib import pyplot as plt
from c04_linear_algebra import vector_mean


# initialize
#invecs = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]

invecs = [[-14,-5],[12.5,12.5],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[10,14.5],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]
# Clustering the data with k=3
random.seed(0)

"""
KMeans Clustering
"""
clusterer = KMeans(3)
assignments = clusterer.train(invecs)
print(clusterer.means)
colors = ['r','g','b']
markers = ['D', 'o', '*']
plt.figure()
for i in range(clusterer.k):
    xys = [invec for invec, assignment in zip(invecs, assignments) if assignment == i]
    plt.scatter([xy[0] for xy in xys], [xy[1] for xy in xys], color = colors[i], marker = markers[i])
plt.xlabel("blocks east of city center")
plt.ylabel("blocks north of city center")
plt.title("User locations -- 3 clusters")

# Clustering the data with k=2
#random.seed(0)
clusterer = KMeans(2)
assignments = clusterer.train(invecs)
print(clusterer.means)
colors = ['r','b','g']
markers = ['D', '*', 'o']
plt.figure()
for i in range(clusterer.k):
    xys = [invec for invec, assignment in zip(invecs, assignments) if assignment == i]
    plt.scatter([xy[0] for xy in xys], [xy[1] for xy in xys], color = colors[i], marker = markers[i])
plt.xlabel("blocks east of city center")
plt.ylabel("blocks north of city center")
plt.title("User locations -- 2 clusters")

# Choosing k
ks = range(1,len(invecs)+1)
sce = [squared_clustering_errors(invecs, k) for k in ks]
plt.figure()
plt.plot(ks, sce)
plt.ylabel("total squared error")
plt.xlabel("k")
plt.title("Total Error vs. # of clusters")

"""
Clustering colors in an image (to reduce number of colors in the image)
"""
imvecs, cc_imvecs = recolor(r"/Users/loanvo/Downloads/fall2015_ss.jpg", 5)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(imvecs)
plt.subplot(1,2,2)
plt.imshow(cc_imvecs)
plt.axis("off")


"""
Bottom-up Hierarchical Clustering
"""
# Distance between two clusters is the MIN distance between elements of the two clusters
base_cluster = bottom_up_cluster(invecs, min)
print(base_cluster_string(base_cluster))
k = 3
colors = ['r', 'g', 'b']
markers = ['D', 'o', '*']
k_clusters = generate_clusters(base_cluster, k)
k_cluster_values = [get_values(k_cluster) for k_cluster in k_clusters]
plt.figure()
for i, c, m, k_cluster_value in zip(list(range(1,k+1)), colors, markers, k_cluster_values):
    x, y = zip(*k_cluster_value)
    plt.scatter(x, y, color = c, marker = m)
    cluster_mean = vector_mean(*k_cluster_value)
    plt.plot(cluster_mean[0], cluster_mean[1], marker='$' + str(i) + '$', color='black')
    
plt.title("User Locations -- 3 Bottom-Up Clusters, Min")
plt.xlabel("blocks east of city center")
plt.ylabel("blocks north of city center")

# Distance between two clusters is the MAX distance between elements of the two clusters
base_cluster = bottom_up_cluster(invecs, max)
print(base_cluster_string(base_cluster))
k = 3
colors = ['r', 'g', 'b']
markers = ['D', 'o', '*']
k_clusters = generate_clusters(base_cluster, k)
k_cluster_values = [get_values(k_cluster) for k_cluster in k_clusters]
plt.figure()
for i, c, m, k_cluster_value in zip(list(range(1,k+1)), colors, markers, k_cluster_values):
    x, y = zip(*k_cluster_value)
    plt.scatter(x, y, color = c, marker = m)
    cluster_mean = vector_mean(*k_cluster_value)
    plt.plot(cluster_mean[0], cluster_mean[1], marker='$' + str(i) + '$', color='black')
    
plt.title("User Locations -- 3 Bottom-Up Clusters, Max")
plt.xlabel("blocks east of city center")
plt.ylabel("blocks north of city center")

plt.show()