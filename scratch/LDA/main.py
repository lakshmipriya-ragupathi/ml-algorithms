import numpy as np
import pandas as pd

def class_means(X_data, y_labels):
    unique_labels = np.unique(y_labels)
    num_labels = len(unique_labels)
    n_samples, n_features = X_data.shape

    class_means = np.zeros((num_labels, n_features))
    for i, label in enumerate(unique_labels):
        X_label = X_data[y_labels == label]
        class_means[i, :] = np.mean(X_label, axis=0)

    return class_means

def within_class_scatter(X_data, y_labels, class_means):
    unique_labels = np.unique(y_labels)
    n_features = X_data.shape[1]

    within_scatter = np.zeros((n_features, n_features))
    for i, label in enumerate(unique_labels):
        X_label = X_data[y_labels == label]
        X_label_mean = X_label - class_means[i]
        within_scatter += np.dot(X_label_mean.T, X_label_mean)

    return within_scatter

def between_class_scatter(X_data, y_labels, class_means):
    unique_labels = np.unique(y_labels)
    n_features = X_data.shape[1]

    between_scatter = np.zeros((n_features, n_features))
    total_mean = np.mean(X_data, axis=0)
    for i, label in enumerate(unique_labels):
        n_label = X_data[y_labels == label].shape[0]
        class_mean_diff = class_means[i] - total_mean
        between_scatter += n_label * np.outer(class_mean_diff, class_mean_diff)

    return between_scatter

def sort_eigenvalues(within_scatter, between_scatter):
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(within_scatter).dot(between_scatter))
    eigenvalue_idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[eigenvalue_idx]
    sorted_eigenvectors = eigenvectors[:, eigenvalue_idx]

    return sorted_eigenvalues, sorted_eigenvectors

def projection_matrix(eigenvectors, components):
    return eigenvectors[:, :components]

def mahalanobis_distance(test_sample, train_data, num_samples):
    mean_value = np.mean(train_data, axis=0)
    mean_duplicate = np.tile(mean_value, (num_samples, 1))
    difference = train_data - mean_duplicate
    covariance_matrix = (np.transpose(difference) @ difference) / (num_samples - 1)
    return np.sqrt((test_sample - mean_value) @ np.linalg.inv(covariance_matrix) @ np.transpose(test_sample - mean_value))

data = pd.read_csv("face.csv")
print(data.shape)
X_data = data.iloc[:, 0:-1].values
y_labels = data.iloc[:, -1].values
print(X_data.shape)
print(y_labels.shape)

class_means = class_means(X_data, y_labels)
within_scatter = within_class_scatter(X_data, y_labels, class_means)
between_scatter = between_class_scatter(X_data, y_labels, class_means)
sorted_eigenvalues, sorted_eigenvectors = sort_eigenvalues(within_scatter, between_scatter)

components = 150
projection_matrix = projection_matrix(sorted_eigenvectors, components)

print("Original shape:", X_data.shape)
X_lda_data = np.dot(X_data, projection_matrix)
print("LDA shape:", X_lda_data.shape)

test_samples = []
train_data = []
accuracy = 0
print()
for i in range(40):
    test_samples.append(X_lda_data[i * 10])
    train_data.append(X_lda_data[i * 10 + 1:(i + 1) * 10])

for i in range(len(test_samples)):
    print(f"Sample : {i}")
    print("Actual class :", i)
    distances = []
    for trainer in train_data:
        distances.append(mahalanobis_distance(test_samples[i], trainer, 9))
    predicted_class = distances.index(min(distances))
    print(f"Predicted Class : {predicted_class}")
    if predicted_class == i:
        accuracy += 1
    print()
    print()

print("Accuracy :", accuracy / len(test_samples) * 100)
