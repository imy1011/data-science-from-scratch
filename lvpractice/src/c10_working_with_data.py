'''
Created on Jul 7, 2017
By Loan Vo
Original File Path: /Users/loanvo/GitHub/py/joelgrus/joelgrus/src/c10_working_with_data.py
'''


''' WORKING WITH DATA:
1. Exploring your data:
    a. One dimensional data: 
        + compute a few summary statistics
        + create a histogram
    b. Two dimensional data:
        + examine each dimension individually (i.e. step for exploring one dimensional data)
        + do a scatter plot on the data
    c. Many dimensions:
        + Look at their correlation matrix
        + If there AREN'T TOO MANY dimensions, it might be good to make a scatter plot matrix
        
        
2. Cleaning and munging:
    a. Change/parse the data into the right data type (int, float, etc.)
    b. Cleaning up outliers if necessary
    c. Cleaning up by ad hoc investigating: 
    For example, look at the data, and manually remove data doesn't make sense such as year=3014
    or check for missing decimal points, extra zeros, typographical erros, etc.
    

3. Manipulating data: get the target columns/fields, then do some calculation on them, e.g.
group them, find max, find min, etc.
    
    
4. Rescaling:
Sometimes units of the data can be problematic and affect the final conclusion (see example 
on page 132 of Joel Grus' book).
To avoid this issue, sometimes we RESCALE our data so that each dimension has mean 0 and 
standard deviation 1 --> converting each dimension to "standard deviations from the mean", 
i.e., doing: demeaned_original_values/Standard_deviation
--> effectively get rid of the units


5. Dimensionality reduction:
'''


from c08_gradient_descent import maximize_batch, maximize_stochastic
from c05_statistics import correlation, standard_variation
from c04_linear_algebra import shape, make_matrix, get_column,\
    magnitude, dot_product, vector_subtract, vector_sum, scalar_multiply
from functools import partial
import numpy as np
from collections import Counter, defaultdict
from matplotlib import pyplot as plt



'''*****************
1a. Exploring your data: One dimensional data
*****************'''

def plot_histogram_of_1dim_data(x, num_bins = 10, fig_title = "Histogram"):
    max_x = max(x)
    min_x = min(x)
    delta_x = (max_x - min_x)/num_bins
    bin_x = Counter(int((x_i-min_x)/delta_x) for x_i in x)
    x_axis = [i*delta_x + min_x for i in range(num_bins)]
    x_ticks = [(i+.5)*delta_x + min_x for i in range(num_bins)]
    y_axis = [bin_x[i] if bin_x[i] else 0 for i in range(num_bins)] # hasn't yet include the last bin (which always has 1 element: the maximum value of x)
    y_axis[-1] += 1 #the last bin only include x's maximum value and hasn't been included into y_axis 
    plt.figure()
    plt.bar(x_axis, y_axis, width = delta_x)
    plt.xlim(min_x-delta_x, max_x + delta_x)
    plt.xticks(x_ticks)
    plt.title(fig_title)
    
def plot_histogram(x, bin_size, fig_title, ax = None): #this function requires bin size instead of number of bins
    binned_x = [bin_size*(x_i//bin_size) for x_i in x]
    bin_counts = Counter(binned_x_i for binned_x_i in binned_x)
    if ax is None:
        plt.figure()
        ax = plt.axes()
    ax.bar(bin_counts.keys(), bin_counts.values(), bin_size)
    plt.title(fig_title)





'''*****************
1c. Exploring your data: multi-dimensional data
*****************'''
def correlation_matrix(data):
    """Return a correlation matrix each element (i,j) of which is the correlation of data in column ith and column jth"""
    _, num_cols = shape(data)
    def matrix_entry(i,j):
        return correlation(get_column(data,i), get_column(data, j))
    return make_matrix(num_cols, num_cols, matrix_entry)


def plot_pairwise_scatterplot(A):
    _, num_cols = shape(A)
    _, ax = plt.subplots(num_cols,num_cols)
    for i in range(num_cols):
        for j in range(num_cols):
            if i == j:
                col_i = get_column(A, i)
                plot_histogram(col_i, (max(col_i)-min(col_i))/20, 'hist of col {:1d}'.format(i), ax[i][j])
                #ax[i][j].annotate("series {}".format(i), xy = (.5, .5), xycoords = "axes fraction" , ha="center", va="center") 
            elif i<j:
                col_i = get_column(A, i)
                col_j = get_column(A, j)
                ax[i][j].scatter(col_i, col_j)
                ax[i][j].annotate("corr={:.2f}".format(correlation(col_i, col_j)), xy=(3, 1), xytext=(0.8, 0.95), xycoords='axes fraction', ha="right", va="top") 
            else:
                ax[i][j].xaxis.set_visible(False)
                       



'''*****************
2. Cleaning and Munging
*****************'''
# Cleaning and parsing data read from csv.reader
def try_or_none(f):
    def f_or_none(x):
        try: return f(x)
        except: return None
    return f_or_none

def parse_row(input_row, parsers):
    return [try_or_none(parser)(ele) if parser is not None else ele 
            for ele, parser in zip(input_row, parsers)]
def parse_rows_with(reader, parsers):
    for row in reader:
        yield parse_row(row, parsers)


# Cleaning and parsing data read from csv.DictReader
def try_parse_field(fieldname, val, dict_parsers):
    parser = dict_parsers.get(fieldname)
    return try_or_none(parser)(val) if parser is not None else val
def parse_dict(dict_row, dict_parsers):
    return {fieldname: try_parse_field(fieldname, val, dict_parsers) 
            for fieldname, val in dict_row.items()}

def parse_dict_with(csv_dict_reader, dict_parsers):
    for dict_row in csv_dict_reader:
        yield parse_dict(dict_row, dict_parsers)




'''*****************
3. Manipulating data
*****************'''



# turn a list of dicts into the list of fieldname values, i.e., return the fieldname column   
def pluck(fieldname, rows):
    return [row[fieldname] for row in rows]

# group rows by provided fieldname and apply a function to each group
# the function should expect input a dict input
# if function is not provided, group_by will return grouped data
def group_by(fieldname, rows, transform_func = None):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[fieldname]].append(row)
    if transform_func is None:
        return grouped
    else:
        return {key: transform_func(rows_within_same_group) 
                for key, rows_within_same_group in grouped.items()}



# applying the group_by function to get one-day percent changes 

def day_over_day_changes(grouped_rows):
    # sorted rows (within the same group)
    sorted_grouped_rows = sorted(grouped_rows, key = lambda x: x["date"], reverse = True)
    return [{"symbol": today_price["symbol"],
            "date": today_price["date"],
            "change": today_price["closing_price"]/yesterday_price["closing_price"]-1}
            for today_price, yesterday_price in zip(sorted_grouped_rows[:-1], sorted_grouped_rows[1:])]

'''*****************
4. Rescaling
*****************'''
def scale(data_matrix):
    _, num_cols = shape(data_matrix)
    mean_each_col = [np.mean(get_column(data_matrix, j)) for j in range(num_cols)]
    std_each_col = [standard_variation(get_column(data_matrix, j)) for j in range(num_cols)]
    return mean_each_col, std_each_col
def rescale(data_matrix):
    mean_each_col, std_each_col = scale(data_matrix)
    num_rows, num_cols = shape(data_matrix)
    def rescale_each_element(i, j):
        return (data_matrix[i][j]-mean_each_col[j])/std_each_col[j] if std_each_col[j]>0 else 0
    return make_matrix(num_rows, num_cols, rescale_each_element)



'''*****************
5. Dimensionality reduction
*****************'''
def de_mean_matrix(A):
    """returns the result of subtracting from every value in A the mean
    value of its column. the resulting matrix has mean 0 in every column"""
    num_rows, num_cols = shape(A)
    column_means, _ = scale(A)
    return make_matrix(num_rows, num_cols, lambda i,j: A[i][j]-column_means[j])

def normalize_matrix(A):
    num_rows, num_cols = shape(A)
    column_means, column_stds = scale(A)
    return [[(A[i][j]-column_means[j])/column_stds[j] 
             for j in range(num_cols)] for i in range(num_rows)]

def direction(w):
    """"normalize a vector so that it is a directional vection"""
    mag_w = magnitude(w)
    return [w_i/mag_w for w_i in w]

def directional_variance_i(x_i, w):
    """projection of x_i on the direction determined by w"""
    w = direction(w)
    return dot_product(x_i, w)**2

def directional_variance(X, w):
    """the variance (~ sum as data is de-meaned) of the data in the direction determined w"""
    w = direction(w)#in Joel's book, the author does not normalize w 
                    # during the optimization process and Loan thinks that 
                    # lacking of normalization would lose the constraint of the
                    # optimization problem and would make cost function explodes to inf 
    return sum(directional_variance_i(x_i,w) for x_i in X)

def directional_variance_gradient_i(x_i, w):
    """the contribution of row x_i to the gradient of
    the direction-w variance"""
    w = direction(w)
    project_length = dot_product(x_i, w)
    return [2*x_i_j*project_length for x_i_j in x_i]

def directional_variance_gradient(X, w):
    w = direction(w) #in Joel's book, the author does not normalize w 
                    # during the optimization process and Loan thinks that 
                    # lacking of normalization would lose the constraint of the
                    # optimization problem and would make cost function explodes to inf
    return vector_sum(*[directional_variance_gradient_i(x_i,w) for x_i in X])
    
def first_principal_component(X):
    """The first principal component is just the direction that maximizes 
    the directional_variance function"""
    guess = direction([1 for _ in X[0]])
    unscaled_maximizer = maximize_batch(
        partial(directional_variance, X), # a function of w
        partial(directional_variance_gradient,X), # a function of w
        guess
        )
    return direction(unscaled_maximizer)

def project(v, w):
    """return the projection of v onto the direction w"""
    return scalar_multiply(dot_product(v, w), w)

#If we want to find further components, we first remove the projections 
#from the data:
def remove_projection_from_vector(v, w):
    """projects v onto w and subtracts the result from v"""
    return vector_subtract(v,project(v,w))

def remove_projection(X, w):
    """for each row of X
    projects the row onto w, and subtracts the result from the row"""
    return [remove_projection_from_vector(xi, w) for xi in X]

"""At that point, we can find the next principal component by repeating 
the process on the result of remove_projection """
def principal_component_analysis(X, num_components):
    components = []
    for _ in range(num_components):
        component = first_principal_component(X)
        X = remove_projection(X, component)
        components.append(component)
    return components 

"""
Instead of using gradient descent, we use stochastic gradient descent to
search for principal axes
"""
def first_principal_component_with_stochastic_gradient_decent(X):
    guess = direction([1 for _ in X[0]])
    unscaled_maximizer = maximize_stochastic(
        directional_variance_i, 
        directional_variance_gradient_i,
        X, 
        guess)
    return direction(unscaled_maximizer)

def principal_component_analysis_with_stochastic_gradient_decent(X, num_components):
    components = []
    for _ in range(num_components):
        component = first_principal_component_with_stochastic_gradient_decent(X)
        X = remove_projection(X, component)
        components.append(component)
    return components 

"""We can then transform our data into the lower-dimensional space spanned 
by the components"""
def transform_vector(v, components):
    num_rows, num_cols = shape(components)
    if num_rows == 1 or num_cols == 1 : #if there is only one principle axis
        return dot_product(v, components)
    else: #if there is more than one principle axis
        return [dot_product(v, component) for component in components]
def transform(X, components):    
    return [transform_vector(x_i, components) for x_i in X]


    
    