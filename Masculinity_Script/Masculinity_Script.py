
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# turning the csv into a dataframe
survey = pd.read_csv("masculinity.csv")

# examining certain general aspects of the dataframe
# print(survey.head())
# print(survey.info())
# print(survey.columns)
# print(survey["q0007_0001"].value_counts())

# mapping the data and giving responses numerical values where linear progression is obvious
# using question 7 since this question is about male lifestyle, could use other questions in cojunction
# with this one but want to understand the data a bit better
cols_to_map = ["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004","q0007_0005", "q0007_0006", "q0007_0007", "q0007_0008", "q0007_0009","q0007_0010", "q0007_0011"]

for col in cols_to_map:
    survey[col] = survey[col].map({"Never, and not open to it": 0, "Never, but open to it": 1, "Rarely": 2, "Sometimes": 3, "Often": 4})

# testing for the number of responses for each category in question 7
# print(survey['q0007_0001'].value_counts())

# plotting some of the data 
plt.scatter(survey['q0007_0001'], survey['q0007_0002'], alpha = 0.1)
plt.xlabel('Ask a Friend For Professional Advice')
plt.ylabel('Ask a Friend For Personal Advice')
# plt.show()

# creating a KMeans Model on subquestions from question 7

# the first 4 questions are tradtionally feminine activities and the other are tradtional male activities
# would be interesting if the clusters split into two distinct clusters of data
rows_to_cluster = survey.dropna(subset = (["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004","q0007_0005", "q0007_0008", "q0007_0009"]))

classifier = KMeans(n_clusters = 2)
classifier.fit(rows_to_cluster[["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004","q0007_0005", "q0007_0008", "q0007_0009"]])

# initial cluster centers
# print(classifier.cluster_centers_)

# separating and investigating the cluster members
cluster_zero_indices = []
cluster_one_indices = []

for label in range(len(classifier.labels_)):
    if classifier.labels_[label] == 0:
        cluster_zero_indices.append(label)
    elif classifier.labels_[label] == 1:
        cluster_one_indices.append(label)

# checking if the indicies of the clusters individual data separated properly
# print(cluster_zero_indices)

cluster_zero_df = rows_to_cluster.iloc[cluster_zero_indices]
cluster_one_df = rows_to_cluster.iloc[cluster_one_indices]

# checking the age distribution between the two clusters, numbers are shown as a percentage of people in each cluster
print(cluster_zero_df['age3'].value_counts()/len(cluster_zero_df))
print(cluster_one_df['age3'].value_counts()/len(cluster_one_df))

# checking the education distribution between the two clusters, numbers are shown as a percentage of people in each cluster
print(cluster_zero_df['educ4'].value_counts()/len(cluster_zero_df))
print(cluster_one_df['educ4'].value_counts()/len(cluster_one_df))

# it looks like the distribution of each cluster is by education and not by age hence
# people who answered these questions are split by their level of education and not by their
# masculine or feminine categories








