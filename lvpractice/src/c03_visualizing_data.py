'''
Created on July 1, 2017

@author: loanvo
'''
import numpy as np

'''
matplotlib: pyplot, display, show, savefig
'''
from matplotlib import pyplot as plt
years = list(range(1950,2011,10))
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

#create a line chart, years on x-axis, gdp on y-axis
plt.figure()
plt.plot(years, gdp, color='green',marker='o',linestyle='solid')

#add a title
plt.title("Nomial GDP")
#add a label to the y-axis
plt.ylabel("Billions of $")


'''
Bar charts
'''
movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

#bars are by default width .8, so we'll add .1 to the left coordinates
# so that each bar is centered
xs = [i + .1 for i, _ in enumerate(movies)]

#plot bars with left x-coordinates [xs], heights [num_oscars]
plt.figure()
plt.bar(xs, num_oscars)

plt.ylabel("# of Academy Awards")
plt.title("My Favorite Movies")

# label x-axis with movie names at bar centers
plt.xticks(np.array(xs) + .5, movies)


# bar chart: good for plotting histograms of bucketed numeric values (in order to visually explore how the values are distributed)
grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]
grade_bin10 = [grade//10*10 for grade in grades]
from collections import Counter
histogram_with_bin10 = Counter(grade_bin10)
plt.figure()
plt.bar(left=[k - 4 for k in histogram_with_bin10.keys()], height=histogram_with_bin10.values(), width=8)
plt.ylabel("# of Students")
plt.ylim(0,5)
plt.xlabel("Decile")
plt.xlim(-5, 105)
plt.xticks([])
plt.title("Distribution of Exam 1 Grades")

#bar chart can mislead easily when y-axis not to start at 0
mentions = [500, 505]
years = [2013, 2014]
#
plt.subplot(121)
plt.bar([y - .4 for y in years],mentions)
plt.axis([2012.5, 2014.5, 499, 506]) #misleading y-axis only shows the part above 500
plt.xticks(years)
plt.ticklabel_format(useOffset=False) #if we don't do this the display on x-axis would be 0 & 1 with a +2.013e3 in the corner)
plt.ylabel("$ of times I heard someone say 'data science'")
plt.title("Look at the 'Huge' increase!")
#
plt.subplot(122)
plt.bar([y - .4 for y in years],mentions)
plt.xlim(2012.5, 2014.5)
plt.xticks(years)
plt.ticklabel_format(useOffset=False) #if we don't do this the display on x-axis would be 0 & 1 with a +2.013e3 in the corner)
plt.ylabel("$ of times I heard someone say 'data science'")
plt.title("Not So Huge Anymore")

'''
Line charts: good for illustrating trends
'''
variance = [1, 2, 4, 8, 16, 32, 64, 128, 256]
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2,1]
total_error = [x+y for x, y in zip(variance, bias_squared)]
xs = [i for i, _ in enumerate(variance)]

plt.figure()
plt.plot(xs, variance, 'g', label = "variance")
plt.plot(xs, bias_squared, 'r-.', label = "bias^2")
plt.plot(xs, total_error, 'b:', label = "total error")
plt.title('The Bias-Variance Tradeoff')
plt.xlabel('model complexity')
plt.legend()


'''
Scatter plots: relationship between two paired sets of data
'''

friends = [70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.figure()
plt.scatter(friends, minutes)
plt.title('Daily Miutes vs Number of friends')
plt.ylabel('daily minutes spent on the site')
plt.xlabel('# of friends')
plt.axis([58, 74, 80, 240])
#label each point
for l, f, m in zip(labels, friends, minutes):
    plt.annotate(l,xy=(f,m),xytext=(f+.2,m-.2))
    
# if you're scattering comparable variables, you might get a misleading picture if you let matplotlib choose the scale
test_1_grades = [99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]

plt.figure()
plt.subplot(121)
plt.scatter(test_1_grades, test_2_grades)
plt.axis([75, 100, 50, 110])
plt.xlabel("test 1 grade")
plt.ylabel("test 2 grade")
plt.title("Axes Aren't Comparable")

plt.subplot(122)
plt.scatter(test_1_grades, test_2_grades)
plt.axis([50, 120, 50, 120])
plt.xlabel("test 1 grade")
plt.ylabel("test 2 grade")
plt.title("Axes Are Comparable")

plt.show()