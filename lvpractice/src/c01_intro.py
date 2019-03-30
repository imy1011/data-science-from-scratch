'''
Created on Jun 18, 2017

@author: loanvo
'''
from collections import defaultdict, Counter
from numpy import nan, average

users = [
    {"id": 0, "name": "Hero"},
    {"id": 1, "name": "Dunn"},
    {"id": 2, "name": "Sue"},
    {"id": 3, "name": "Chi"},
    {"id": 4, "name": "Thor"},
    {"id": 5, "name": "Clive"},
    {"id": 6, "name": "Hicks"},
    {"id": 7, "name": "Devin"},
    {"id": 8, "name": "Kate"},
    {"id": 9, "name": "Klein"}
    ]
friendships = [(0,1), (0,2), (1,2), (1,3), (2,3), (3,4), (4,5), (5,6), (5,7), (6,8), (7,8), (8,9)]

#For each user, add "friends" info which shows his friends' ids
'''
# Although this code seems longer, it doesn't assume that user id is the same as their index in the list users
for user in users:
    friends = []
    for friendship in friendships:
        if user["id"] == friendship[0]:
            friends.append(friendship[1])
        if user["id"] == friendship[1]:
            friends.append(friendship[0])
    user["friends"] = friends
print(users)
'''   

# shorter codes but assume that index in the list of users is the same as user's id
for user in users:
    user["friends"] = []
for i,j in friendships:
    users[i]["friends"].append(users[j]["id"])
    users[j]["friends"].append(users[i]["id"])
print("Users:",users)


# Find the average number of connections/friends each person has
numOfPeople = len(users)
totalNumOfConnections = sum([len(user["friends"]) for user in users])
print("Average number of connections", totalNumOfConnections / numOfPeople) 


# listing all original info (friends, id, name) of each user sorted by his number of friends
print("Sorted users:",sorted(users,key=lambda x: len(x["friends"]),reverse = True))

# listing only id and number of friends of each user sorted by his number of friends
numOfFriendsById = [(user["id"],len(user["friends"])) for user in users]
print("Sorted users:",sorted(numOfFriendsById,key=lambda x: x[1],reverse = True))

# find a list of all users who are friends or friends of friends of a given user
def all_friends_or_friends_of_friends_ids(user):
    afofof = set()
    for x in user["friends"]:
        afofof.add(x)
        for y in users[x]["friends"]:
            afofof.add(y)
    afofof.discard(user["id"])
    return afofof
print("All friends or friends of friends of a user:",all_friends_or_friends_of_friends_ids(users[0]))

# determine if a pair of users are friends of each other
def friend(user1,user2):
    return user2["id"] in user1["friends"]
print("Are they friends?",friend(users[0],users[1]))

# find number of mutual friends between a given user and his non-friend users
def numOfMutualFriendsBetweenAGivenUserAndHisNonFriendUser(user):
    outputKeys = []
    outputValues = []
    for otherUser in users:
        if otherUser["id"] != user["id"] and not friend(otherUser,user) and otherUser["id"] in all_friends_or_friends_of_friends_ids(user):
                #print("-----",user["id"],"and",otherUser["id"])
                temp = len(set(user["friends"]).intersection(otherUser["friends"]))
                if temp != 0:
                    outputKeys.append(otherUser["id"])
                    outputValues.append(temp)
    return dict(zip(outputKeys,outputValues))

userId = 0
print("Number of mutual friends between user:", userId, "and other people who are not his friends:",numOfMutualFriendsBetweenAGivenUserAndHisNonFriendUser(users[0]))

interest = [
    (0,"Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
    (0,"Spark"), (0, "Storm"), (0, "Cassandra"),
    (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
    (1, "Postgres"), (1,"Python"),
    (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
    (3, "statistics"), (3, "regression"), (3, "probability"),
    (4, "machine leanring"), (4, "regression"), (4, "decision trees"),
    (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
    (5, "Haskell"), (5, "programming languages"), (6, "statistics"),
    (6, "probability"), (6, "mathematics"), (6,"theory"),
    (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
    (7, "neural network"), (8, "neural network"), (9, "deep learning"),
    (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
    (9, "Java"), (9, "MapReduce"), (9, "BigData")
    ]
     
# Build an index from interests to users: keys are interests, values are list of user_id with that interest
user_ids_by_interest = defaultdict(list)
for userId, eachInterest in interest:
    user_ids_by_interest[eachInterest].append(userId)
print(user_ids_by_interest)
# Build an index from users to interest: keys are users, values are list of interest for that user_id
interests_by_user_id = defaultdict(list)
for userId, eachInterest in interest:
    interests_by_user_id[userId].append(eachInterest)
print(interests_by_user_id)
# Find who has the most interest in common with a given user
common_interest_by_user_id = defaultdict(list)
def most_common_interests_with(user):
    for interest in interests_by_user_id[user]:
        for user_with_same_interest in user_ids_by_interest[interest]:
            if user_with_same_interest != user:
                common_interest_by_user_id[user_with_same_interest].append(interest)
    user_id_with_most_interest_in_common = nan
    number_of_common_interest = 0
    for user_id in common_interest_by_user_id:
        if len(common_interest_by_user_id) >= number_of_common_interest:
            user_id_with_most_interest_in_common = user_id 
            number_of_common_interest = len(common_interest_by_user_id) 
    return user_id_with_most_interest_in_common
print(most_common_interests_with(0))

# Salaries and experience
salaries_and_tenures = [(83000, 8.7), (88000, 8.1), (48000, 0.7), (76000, 6), (69000, 6.5), (76000, 7.5), (60000, 2.5), (83000, 10), (48000, 1.9), (63000, 4.2)]
# put tenure into "tenure bucket"/groups
def tenure_bucket(tenure):
    if tenure < 2:
        return "less than two"
    elif tenure < 5:
        return "between two and five"
    else:
        return "more than five"
# group salaries by tenure period bucket
salary_by_tenure_bucket = defaultdict(list)
for salary, tenure in salaries_and_tenures:
    salary_by_tenure_bucket[tenure_bucket(tenure)].append(salary)
print(salary_by_tenure_bucket)


# computing the average salary for each tenure group:
average_salary_by_bucket = {tenure: sum(salaries)/len(salaries) for tenure, salaries in salary_by_tenure_bucket.items()}
print(average_salary_by_bucket)
# find the most popurlar interest topic (~each different word) in the above interest list
word_and_count = Counter(word for dummy, eachinterest in interest for word in eachinterest.lower().split())
print(word_and_count)
