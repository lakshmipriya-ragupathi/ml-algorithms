import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


'''
3. In this question, you will implement the Minimum Spanning Tree (MST) approach to cluster the Iris dataset
using only the sepal length and sepal width features. 
Your task is to write a Python program that performs the following steps:

Load the Iris dataset.

Extract the sepal length and sepal width columns.

Calculate the pairwise Euclidean distances between data points using the selected features.

Build the Minimum Spanning Tree using Prim's/Kruskal's algorithm.

Identify the two longest edges in the MST and remove them to create three clusters.

Visualize the Minimum Spanning Tree with identified clusters.


'''


#load the data:
df = pd.read_csv('iris.csv')
print(df.info())
print(df.head())

#label the classes as 0, 1 and 2
df['Target'] = np.where(df['Species'] == 'Iris-setosa', 0, np.where(df['Species'] == 'Iris-versicolor', 1, 2))
print(df.head())                        




#extract sepal length and sepal width columns
df = df.drop(['PetalLengthCm', 'PetalWidthCm'], axis = 1)
print(df.info())

#data
x = df.drop(['Id', 'Species', 'Target'], axis = 1)
#labels
y = df.drop(['Id', 'SepalLengthCm', 'SepalWidthCm', 'Species'], axis = 1)



#Calculate the pairwise Euclidean distances between data points using the selected features.
from scipy.spatial.distance import pdist, squareform
pair_wise_distance = pdist(x)
print(pair_wise_distance)
#convert single row into square matrix
pair_wise_distance_matrix = squareform(pair_wise_distance)



#Build the Minimum Spanning Tree using Prim's/Kruskal's algorithm.
edges = [(i, j, pair_wise_distance_matrix[i, j]) for i in range(x.shape[0]) for j in range(i + 1, x.shape[0])]

class disjoint_set:
    def __init__(self,n):
        self.parent = [i for i in range(n)]
        self.rank = [0] * n

    def find_parent(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find_parent(self.parent[x])
        return self.parent[x]
    def set_union(self, a, b):
        root_a = self.find_parent(a)
        root_b = self.find_parent(b)

        if root_a != root_b:
            if self.rank[root_a] < self.rank[root_b]:
                self.parent[root_a] = root_b
            elif self.rank[root_a] > self.rank[root_b]:
                self.parent[root_b] = root_a
            else:
                self.parent[root_b] = root_a
                self.rank[root_a] += 1

#kruskals algorithm:

edges.sort(key = lambda x : x[2])
n = len(edges)
mstree = set()
disjoint_union = disjoint_set(n)

for edge in edges:
    u,v, weight = edge
    if disjoint_union.find_parent(u) != disjoint_union.find_parent(v):
        mstree.add(edge)
        disjoint_union.set_union(u,v)



#Identify the two longest edges in the MST 
#and remove them to create three clusters.

max_edges = sorted(mstree, key=lambda x: x[2], reverse=True)[:2]
mstree = [edge for edge in mstree if edge not in max_edges]



#Visualize the Minimum Spanning Tree with identified clusters.
plt.figure(figsize=(15, 9))
plt.xlim(4, 8)
plt.ylim(1.5, 4.5)

plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')

for target, marker, label in zip([0, 1, 2], ['o', '*', '+'], ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']):
    subset = x[y['Target'] == target]
    plt.scatter(subset.SepalLengthCm, subset.SepalWidthCm, s=12, label=label, marker=marker)

plt.legend()

for edge in mstree:
    x_values = [x.iloc[edge[0]].SepalLengthCm, x.iloc[edge[1]].SepalLengthCm]
    y_values = [x.iloc[edge[0]].SepalWidthCm, x.iloc[edge[1]].SepalWidthCm]
    plt.plot(x_values, y_values, linestyle='-', color='greenyellow', alpha=0.7)

plt.show()