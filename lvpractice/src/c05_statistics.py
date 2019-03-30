'''
Created on July 2, 2017

@author: loanvo
'''

import numpy as np
import random
from collections import Counter

'''
1. Number of data points: len(data)
2. Largest value: max(data)
3. Smallest value: min(data)
4. Central Tendencies:
    - mean:
        + Depend on EVERY VALUE in your data 
            --> varies smoothly as data changes
            --> very sensitive to outliers
        + 
    - median: middle-most value if len(data) is odd; or average of two middle-most values if len(data) is even:
        + Does NOT depend on every value in your data
            --> might or might not change at all when one/some data points change values
        + More complicated to calculate: data has to be SORTED first 
        (although there is some efficient algorithm which can avoid sorting) 
    - Quantile (a generalization of the median): which represents the value less than which a certain percentile of data lies
        (median: represent the value less than which 50% of the data lies)
    - Mode: the most-common value[s]
    
5. Dispersion: measures of how spread out data is
    - range: difference between the largest and smallest elements:
        + Like the median, DOES NOT depend on the whole data set
        + Like the mean, suffer from outlier issue
    - variance: how a single variable deviates from its mean: sum of squares of deviation from the mean / (totalNumberOfDataPoints-1)
        + sample variance vs. population one
    - standard deviation: sqrt(variance) : so that the unit of the measurement is the same as the original unit
        + Like the mean, suffer outlier issue
    - interquantile range: difference between the 75th percentile value and the 25th percentile value
        + More robust from outlier issue than range or standard deviation
    
6. Correlation:
    - Covariance: how two variables vary in tandem from their means
        + "large" positive covariance: x tends to be large (or small) when y is large (or small). 
        A "large" negative covariance: x tends to be large (or small) when y is small or large.
        A close-to-zero covariance: no such relationship
        + It is HARD to say how large is large or what counts as a large covariance
        + Its unit can be hard to make sense of
        
    - Correlation: 
        + unitless
        + range from -1 (perfect anti-correlation) to 1 (perfect correlation)
        + Depend on outliner heaviliy
        + only can show LINEAR correlation. 
        --> A correlation of zero indicates that there is NO LINEAR relationship, i.e., there may be other sorts of relationship
        + Correlation is NOT CAUSATION: if x and y are strongly correlated, that might means:
            . x causes y
            . y causes x
            . each causes the other
            . some third factor causes both 
            . it might mean nothing
        --> To feel more confidient about causality, one can conduct RANDOMIZED TRIAL. 
        For example, one can randomly split users into two groups with similar demographics 
        and give on of the groups a slightly different experience --> can often feel pretty good that 
        the difference experiences are causing the different outcomes.
        
7. Simpson's Paradox: correlation can be misleading when confounding variables are ignored. 
The key issue is that correlation is measuring the relationship between two variables all else being equal... 
The only way to avoid this is by knowing your data and by doing what you can to make sure you've checked 
for possible confounding factors.  
'''

def mean(x):
    return sum(x)/len(x)

def median(x):
    """finds the 'middle-most' value of v"""
    sorted_x = sorted(x)
    len_x = len(x)
    if len_x%2 == 0 :
        # if even, return the average of the middle values
        midnum = (sorted_x[int(len_x/2)-1] + sorted_x[int(len_x/2)])/2
    else:
        # if odd, return the middle value
        midnum = sorted_x[int(len_x/2)]
    return midnum


'''
random.seed(10)
num_friends = [ random.choice(range(200)) for _ in range(50) ] # generate number of friends of 50 people. Maximum number of friends a person can have is 200

total_data_points = len(num_friends); print("total number of data points:", total_data_points) 
max_num_friends = max(num_friends); print("largest number of friends:", max_num_friends) 
min_num_friends = min(num_friends); print("smallest number of friends:", min_num_friends) 
mean_num_friends = np.mean(num_friends); print("average number of friends:", mean_num_friends) 
median_num_friends = np.median(num_friends); print("median number of friends:", median_num_friends)
'''
def quantile(x, p): #returns the pth-percentile value in x
    p_index = int(p*len(x))
    return sorted(x)[p_index]
'''
print("10% quantile:", quantile(num_friends, .10)) 
print("25% quantile:", quantile(num_friends, .25)) 
print("50% quantile:", quantile(num_friends, .50)) 
print("75% quantile:", quantile(num_friends, .75)) 
print("90% quantile:", quantile(num_friends, .90)) 
'''

def mode(x):
    c = Counter(x)
    _, most_common_count  = c.most_common(1)[0]
    return [ vi for vi, ci in c.items() if ci == most_common_count ]
#print("Mode value(s):", mode(num_friends))

def data_range(x):
    return max(x) - min(x)
#print("Data range:", data_range(num_friends))

def de_mean(x):
    return x - np.mean(x)
def variance(x):
    return sum(de_mean(x)**2)/(len(x)-1)
#print("Variance:", variance(num_friends))

def standard_variation(x):
    return np.sqrt(variance(x))
#print("Standard deviation:", standard_variation(num_friends))

def interquantile_range(x):
    return quantile(x, .75) - quantile(x, .25)
#print("Interquantile:", interquantile_range(num_friends))

# Covariance:
def covar(x, y):
    return np.dot(x - np.mean(x), y - np.mean(y))/(len(x) - 1)

def correlation(x, y):
    std_x = standard_variation(x)
    std_y = standard_variation(y)
    if std_x > 0 and std_y > 0:
        return covar(x, y) / (std_x * std_y)
    else:
        return 0

# Correlation with/without outliers.

