import pandas as pd
import numpy as np

# Standardize the data
def standardize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data


# Calculate the covariance matrix
def calculate_covariance_matrix(data):
    covariance_matrix = np.cov(data, rowvar=False)
    return covariance_matrix


# Calculate eigenvalues and eigenvectors
def calculate_eigenvalues_and_eigenvectors(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    return eigenvalues, eigenvectors


# Sort eigenpairs
def sort_eigenpairs(eigenvalues, eigenvectors):
    eigenpairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]
    eigenpairs.sort(reverse=True, key=lambda x: x[0])
    return eigenpairs


# Select top eigenvectors
def select_top_eigenvectors(eigenpairs, k):
    top_eigenpairs = eigenpairs[:k]
    top_eigenvectors = np.array([eig[1] for eig in top_eigenpairs]).T
    return top_eigenvectors


# Project data onto eigenvectors
def project_data(data, eigenvectors):
    projected_data = np.dot(data, eigenvectors)
    return projected_data


# Determine the number of principal components for a given variance threshold
def select_dimensions_for_variance(eigenvalues, threshold=0.95):
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    cumulative_explained_variance = np.cumsum(explained_variance)
    return np.where(cumulative_explained_variance > threshold)[0][0]


# Calculate the Mahalanobis distance
def mahalanobis(test, train, n):
    mean = np.mean(train, axis=0)
    mean_dup = np.tile(mean, (n, 1))
    z = train - mean_dup
    cov = (np.transpose(z) @ z) / (n - 1)
    return np.sqrt((test - mean) @ np.linalg.inv(cov) @ np.transpose(test - mean))

df = pd.read_csv("./face.csv")
df.pop("target")

standardized_data = standardize_data(df)
print(f'Initial dimensions : {standardized_data.shape}')

# Calculate the covariance matrix
cov_matrix = calculate_covariance_matrix(standardized_data)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = calculate_eigenvalues_and_eigenvectors(cov_matrix)

# Select the number of dimensions for the desired variance threshold
k = select_dimensions_for_variance(eigenvalues, threshold=0.95)

eigenpairs = sort_eigenpairs(eigenvalues, eigenvectors)
top_eigenvectors = select_top_eigenvectors(eigenpairs, k)
reduced_data = project_data(standardized_data, top_eigenvectors)

# Print the reduced dimensions
print(f'Reduced dimensions : {reduced_data.shape}')

# Create test and train sets
test = []
train = []
for i in range(40):
    test.append(reduced_data[i * 10])
    train.append(reduced_data[i * 10 + 1:(i + 1) * 10])

# Print the length of the first train set
print(len(train[0]))

# Iterate over test samples
for i in range(len(test)):
    print(f"Sample : {i}")
    dist = []
    
    # Compute Mahalanobis distance for each training sample
    for trainer in train:
        dist.append(mahalanobis(test[i], trainer, 9))
    
    # Identify the class based on minimum distance
    cls = dist.index(min(dist))
    
    # Print the class
    print(f"Class : {cls}")
