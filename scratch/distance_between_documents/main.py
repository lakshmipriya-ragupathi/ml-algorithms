import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.feature_extraction.text import CountVectorizer
import math
'''
NOTE :- Do NOT use the inbuilt function for calculating KL Distance,
Bhattacharyya Distance and 
Cosine Distance directly,
you can use functions required to build the distances from scratch like inverse of a matrix, etc.


Compare two text files doc1.txt and doc2.txt using cosine distance.

doc1.txt

“ MATLAB is a program for solving engineering and mathematical problems. The basic MATLAB objects are vectors and matrices, so you must be familiar with these before making extensive use of this program.”

doc2.txt

“MATLAB works with essentially one kind of object, a rectangular numerical matrix. Here is some basic information on using MATLAB matrix commands.”

'''
#reading the file and storing the contents in text1 and text2
with open('doc1.txt', 'r') as file1, open('doc2.txt', 'r') as file2:
    text1 = file1.read()
    text2 = file2.read()

# Create a list of strings for the CountVectorizer
corpus = [text1, text2]

# Initialize the CountVectorizer
vectoriser = CountVectorizer()

# Fit and transform the CountVectorizer on the corpus
vectorizer = vectoriser.fit_transform(corpus)

# The resulting vectors are stored as a sparse matrix, you can convert it to dense format if needed
vector1 = vectorizer[0].toarray()
vector2 = vectorizer[1].toarray()

'''
# Print the vectors
print("Vector for doc1:")
print(vector1)
print("******************************************************")
print("Vector for doc2:")
print(vector2)
print("******************************************************")
'''

product = 0
norm1 = 0
norm2 = 0
for i in range(len(vector1)):
    product = product + (vector1[i]*vector2[i])
    norm1 = norm1 + (vector1[i]*vector1[i])
    norm2 = norm2 + (vector2[i]*vector2[i])

p = 0
for i in range(len(product)):
    p = p + product[i]


n1 = 0
for i in range(len(norm1)):
    n1 = n1 + norm1[i]

n2 = 0
for i in range(len(norm2)):
    n2 = n2 + norm2[i]

norm_1 = math.sqrt(n1)
norm_2 = math.sqrt(n2)

cos_dist = p / (norm_1 * norm_2)
final_dist = 1 - cos_dist
print("Cosine distance between the two given documents is : ", final_dist)
