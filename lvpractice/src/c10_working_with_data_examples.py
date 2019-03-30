'''
Created on Jul 19, 2017
By Loan Vo
Original File Path: /Users/loanvo/datascience/joelgrus/src/c10_working_with_data_examples.py
'''

import random
import dateutil
from collections import defaultdict
from csv import reader, DictReader
from matplotlib import pyplot as plt
from c10_working_with_data import plot_histogram_of_1dim_data, plot_histogram, \
    plot_pairwise_scatterplot, parse_dict_with, parse_rows_with, group_by, \
    day_over_day_changes, principal_component_analysis, pluck, rescale,\
    transform, scale, de_mean_matrix, normalize_matrix, \
    principal_component_analysis_with_stochastic_gradient_decent,\
    remove_projection
from c05_statistics import correlation
from c04_linear_algebra import matrix_transpose, scalar_multiply, get_column, \
    vector_add, shape



'''*****************
1a. Exploring your data: One dimensional data
*****************'''

# Create sample data sets
# Set the random seed
random.seed(0)
num_of_samples = 1000
# with uniform distribution
uniform_dist_data = [random.random() for _ in range(num_of_samples) ]
# with normal distribution
normal_dist_data = [random.normalvariate(0,1) for _ in range(num_of_samples) ]


for num_bins in [10]:  #[10, 30, 50]
    plot_histogram_of_1dim_data(uniform_dist_data, num_bins, "Histogram of {0} of uniform-dist samples. Bins={1}".format(num_of_samples,num_bins))  
    plot_histogram_of_1dim_data(normal_dist_data, num_bins, "Histogram of {0} of normal-dist samples. Bins={1}".format(num_of_samples,num_bins))  

for bin_size in [.1]: #[10, 30, 50]
    plot_histogram(uniform_dist_data, bin_size,"Histogram of {0} of uniform-dist samples. Bin size={1}".format(num_of_samples, bin_size) )
    plot_histogram(normal_dist_data, bin_size,"Histogram of {0} of normal-dist samples. Bin size={1}".format(num_of_samples, bin_size) )


'''*****************
1b. Exploring your data: Two dimensional data
*****************'''
xs = [random.normalvariate(0,1) for _ in range(1000)]
ys1 = [x + random.normalvariate(0,1)/2 for x in xs]
ys2 = [-x + random.normalvariate(0,1)/2 for x in xs]
plt.figure()
plt.scatter(xs, ys1, marker='.', color='r', label = "ys1. Correlation(xs,ys1)={0:.3f}".format(correlation(xs, ys1)))
plt.scatter(xs, ys2, marker='.', color='b', label = "ys2. Correlation(xs,ys2)={0:.3f}".format(correlation(xs, ys2)))
plt.legend(loc=9)
plt.xlabel('xs')
plt.ylabel('ys')
plt.title('Very Different Joint Distribution')


'''*****************
1c. Exploring your data: multi-dimensional data
*****************'''

num_of_eles = 100
x1 = [random.normalvariate(0,1) for _ in range(num_of_eles)]
data = [[x + random.normalvariate(0,1)/2 for x in x1],
        [-x + random.normalvariate(0,1)/2 for x in x1],
        [random.choice([0, 1]) for _ in x1],
        [random.normalvariate(0,1) for _ in x1]]
data = matrix_transpose(data)

plot_pairwise_scatterplot(data)


'''*****************
2. Cleaning and Munging
*****************'''

csv_list_data = []
with open("/Users/loanvo/datascience/joelgrus/data/comma_delimited_stock_prices.csv",'rt') as f:
    csv_reader = reader(f, delimiter=',')
    for line in parse_rows_with(csv_reader, [dateutil.parser.parse, None, float]):
        csv_list_data.append(line)
print("Row entries with none, n/a, invalid data:")
for row in csv_list_data:
    if any(x is None for x in row):
        print(row)  


print("Parsing data with csv.DictReader:")
#dict_data = defaultdict(list)
dict_data = list()
with open("/Users/loanvo/datascience/joelgrus/data/colon_delimited_stock_prices.txt",'rt') as f:
    csv_dict_reader = DictReader(f, delimiter=':')
    dict_parsers = {"date": dateutil.parser.parse, "symbol": None, "closing_price": float}
    for dict_row in parse_dict_with(csv_dict_reader, dict_parsers):
        #for fieldname, val in dict_row.items():
            #dict_data[fieldname].append(val) 
        dict_data.append(dict_row)
print(dict_data)    


'''*****************
3. Manipulating data
*****************'''

# Find the highest-ever closing price for each stock in dict_data
by_symbol = defaultdict(list)
for row in dict_data:
    by_symbol[row["symbol"]].append([row["closing_price"], row["date"]])
max_price_by_symbol = {symbol: max(val[0] for val in vals) 
                       for symbol, vals in by_symbol.items()}
print("Maximum closing_price of each stock:", max_price_by_symbol)

# applying the group_by function to get the maximum closing_price of each stock.
max_price_by_symbol_from_group_by = group_by('symbol', dict_data, 
                                             lambda rows: max(pluck("closing_price",rows)))
print("Applying group_by() to get maximum closing_price of each stock:", 
      max_price_by_symbol_from_group_by)    

# applying the group_by function to get one-day percent changes 

changes_by_symbol = group_by('symbol', dict_data, day_over_day_changes)
#changes_by_symbol.values(): list of list of dict  --> make them into a single big list
all_changes = [change for list_of_changes_of_each_stock in changes_by_symbol.values()
               for change in list_of_changes_of_each_stock]
print("Largest change over a day:", max(all_changes, key=lambda x: x["change"]))
print("Smallest change over a day:", min(all_changes, key=lambda x: x["change"]))

'''*****************
4. Rescaling
*****************'''
print("Before rescaling:");print(*data,sep='\n')
print("After rescaling:");print(*rescale(data), sep='\n')



'''*****************
5. Dimensionality reduction
*****************'''
'''
The mat_A value in this example is copied from 
http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf
And I can verify my program result with the result in this pdf document
'''
mat_A = [[2.5, 2.4], 
         [.5, .7],
         [2.2, 2.9],
         [1.9, 2.2], 
         [3.1, 3.0],
         [2.3, 2.7],
         [2, 1.6],
         [1, 1.1],
         [1.5, 1.6],
         [1.1, .9]]
col_mean, _ = scale(mat_A)
# plot the data BEFORE de-meaning them
plt.figure()
plt.subplot(231)
plt.scatter(get_column(mat_A, 0), get_column(mat_A, 1), c = 'k', marker = '+')
plt.title("Original data (before demean)")
plt.xlim((-1,4))
plt.ylim((-1,4))
plt.plot([0, 0], [4, -1], 'c:')
plt.plot([4, -1], [0, 0], 'c:')
# demean the data
demean_mat_A = de_mean_matrix(mat_A)
print(*demean_mat_A, sep='\n')
# plot the data AFTER de-meaning them
plt.subplot(232)
plt.scatter(get_column(demean_mat_A, 0), get_column(demean_mat_A, 1), c = 'k', marker = '+')
plt.title("Demean data with principle axes overlayed")
plt.xlim((-2,2))
plt.ylim((-2,2))
# Find the principal axes
components = principal_component_analysis(demean_mat_A, 2)
print("\nPrincipal axes:")
print(*components, sep = "\n")
components_stochastic = \
    principal_component_analysis_with_stochastic_gradient_decent(demean_mat_A, 2)
print("\nPrincipal axes (stochastic gradient descent):")
print(*components_stochastic, sep = "\n")
# add the two principle components on the the demean data scatter plot
x_axis1 = [2*components[0][0], -2*components[0][0]]
y_axis1 = [2*components[0][1], -2*components[0][1]]
x_axis2 = [2*components[1][0], -2*components[1][0]]
y_axis2 = [2*components[1][1], -2*components[1][1]]
plt.plot(x_axis1, y_axis1, 'r--', label = "Principal axis 1")
plt.plot(x_axis2, y_axis2, 'b--', label = "Principal axis 2")
plt.legend(loc=9, prop = {'size': 6})
plt.plot([0, 0], [2, -2], 'c:')
plt.plot([2, -2], [0, 0], 'c:')
# Transform the data into the new coordinates formed from principal axes
# Using two principal axes as new coordinates
transform_mat_2 = transform(demean_mat_A, components)
print("\nTransformed data in the new coordinate formed by TWO principal axes:")
print(*transform_mat_2, sep='\n')
# plot data in new coordinates
plt.subplot(233)
plt.scatter(get_column(transform_mat_2, 0), get_column(transform_mat_2, 1), c = 'k', marker = '+')
plt.title("Transformed data in coordinates \n formed by 2 principal axes")
plt.xlim((-2,2))
plt.ylim((-2,2))
plt.plot([0, 0], [2, -2], 'c:')
plt.plot([2, -2], [0, 0], 'c:')
# using ONE principal axis as new coordinate
transform_mat_1 = transform(demean_mat_A, components[0])
print("\nTransformed data in the new coordinate formed by ONE principle axis:")
print(*transform_mat_1, sep='\n')

# Data after removing its 1st principal component:
A_removed_1st_comp = remove_projection(demean_mat_A, components[0])
plt.subplot(234)
plt.scatter(get_column(A_removed_1st_comp,0), 
            get_column(A_removed_1st_comp,1),
            c = 'b', marker = '+')
plt.title("Data after removed \n its first principal component")

# Restoring original data (their approximation reconstruction) when used only the 1st principle component
restored_mat_from_1st_principle_axis = [vector_add(scalar_multiply(data_i, components[0]),
                                        col_mean) for data_i in transform_mat_1] 
print("\nRestored data from its first principal component:")
print(*restored_mat_from_1st_principle_axis, sep = "\n")
# plot the restored data
plt.subplot(235)
plt.scatter(get_column(restored_mat_from_1st_principle_axis, 0), 
            get_column(restored_mat_from_1st_principle_axis, 1), c = 'r', marker = '.',
            edgecolors = 'r', label = 'restored data')
plt.scatter(get_column(mat_A, 0), get_column(mat_A, 1), c = 'k', marker = '+',
            label = 'original data')
plt.xlim((-1,4))
plt.ylim((-1,4))
plt.plot([0, 0], [4, -1], 'c:')
plt.plot([4, -1], [0, 0], 'c:')
plt.legend(loc = 2, prop = {'size': 7})
plt.title("Original data vs restored ones based on \n their 1st principle component")


'''
3-dim data
Data in this example copied from 
http://www.itl.nist.gov/div898/handbook/pmc/section5/pmc552.htm
'''
print("\n\n\n Finding principal axes of 3d data (3-column matrix)")
mat_A = [[7, 4, 3],
         [4, 1, 8],
         [6, 3, 5],
         [8, 6, 1],
         [8, 5, 7],
         [7, 2, 9],
         [5, 3, 3],
         [9, 5, 8],
         [7, 4, 5],
         [8, 2, 2]]
num_rows, num_cols = shape(mat_A)
# For this data set, we not only demean but also normalize std of each column 
normalized_mat_A = normalize_matrix(mat_A)
"""
# Principal axes of mat_A are also eigenvectors of mat_A's correlation 
# matrix (normalized with num_rows-1)
normalizedMatA = np.array(normalized_mat_A)
correlationMat_normalizedMatA = np.matmul(np.transpose(normalizedMatA), \
    normalizedMatA)/(num_rows-1)
print(correlationMat_normalizedMatA, sep = "\n")
print("\n\n")
"""
# Find the principal axes with batch gradient descent
components_3dData = principal_component_analysis(normalized_mat_A, 3)
print("Principal axes:")
print(*components_3dData,sep = '\n')
# Find the principal axes with stochastic gradient descent
components_stochastic_3dData = \
    principal_component_analysis_with_stochastic_gradient_decent(normalized_mat_A, 3)
print("\nPrincipal axes (stochastic gradient descent):")
print(*components_stochastic_3dData, sep = "\n")



plt.show() 






