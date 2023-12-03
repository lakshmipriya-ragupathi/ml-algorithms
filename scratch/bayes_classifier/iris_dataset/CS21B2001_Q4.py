import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def bayes_boundary(sigma_class1, sigma_class2, mean_class1, mean_class2, colour, class_label):
    W1 = -0.5 * np.linalg.inv(sigma_class1)
    w1 = np.linalg.inv(sigma_class1) @ mean_class1
    w10 = -0.5 * mean_class1.T @ np.linalg.inv(sigma_class1) @ mean_class1 - 0.5 * np.log(
        np.linalg.det(sigma_class1)
    )

    W2 = -0.5 * np.linalg.inv(sigma_class2)
    w2 = np.linalg.inv(sigma_class2) @ mean_class2
    w20 = -0.5 * mean_class2.T @ np.linalg.inv(sigma_class2) @ mean_class2 - 0.5 * np.log(
        np.linalg.det(sigma_class2)
    )

    W = W1 - W2
    w = w1 - w2
    w0 = w10 - w20

    x_vals = np.linspace(-6, 6, 500)
    y_vals = np.linspace(-6, 6, 500)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z = (
        W[0, 0] * X**2
        + W[1, 1] * Y**2
        + (W[1, 0] + W[0, 1]) * X * Y
        + w[0] * X
        + w[1] * Y
        + w0
    )
    print(f"\nDecision boundary of class {class_label}")
    print(f'g({class_label}) = {W[0,0]}x^2 + {W[1,1]}y^2 + {W[1,0]+W[0,1]}xy + {w[0]}x + {w[1]}y + {w0} = 0')
    plt.contour(X, Y, Z, levels=[0], colors=colour)


def bayes_classifier(sigma_class1, sigma_class2, mean_class1, mean_class2, class_label):
    W1 = -0.5 * np.linalg.inv(sigma_class1)
    w1 = np.linalg.inv(sigma_class1) @ mean_class1
    w10 = -0.5 * mean_class1.T @ np.linalg.inv(sigma_class1) @ mean_class1 - 0.5 * np.log(
        np.linalg.det(sigma_class1)
    )

    W2 = -0.5 * np.linalg.inv(sigma_class2)
    w2 = np.linalg.inv(sigma_class2) @ mean_class2
    w20 = -0.5 * mean_class2.T @ np.linalg.inv(sigma_class2) @ mean_class2 - 0.5 * np.log(
        np.linalg.det(sigma_class2)
    )

    W = W1 - W2
    w = w1 - w2
    w0 = w10 - w20

    print(f"\nDecision boundary of class {class_label}")
    print(
        f'g({class_label}) = {W[0,0]}w^2 + {W[1,1]}x^2 + {W[2,2]}y^2 + {W[3,3]}z^2 + \n{W[0,1]+W[1,0]}wx + {W[0,2]+W[2,0]}wy + {W[0,3]+W[3,0]}wz + {W[1,2]+W[2,1]}xy + {W[1,3]+W[3,1]}xz + {W[2,3]+W[3,2]}yz + \n{w[0]}w + {w[1]}x + {w[2]}y + {w[3]}z + \n{w0} = 0'
    )


df = pd.read_csv("iris.csv")
df.drop(columns=["Id", "Species"], inplace=True)

class1_data = df.iloc[0:40, :].to_numpy()
class2_data = df.iloc[50:90, :].to_numpy()
class3_data = df.iloc[100:140, :].to_numpy()

sigma_class1 = np.cov(class1_data, rowvar=False)
sigma_class2 = np.cov(class2_data, rowvar=False)
sigma_class3 = np.cov(class3_data, rowvar=False)

mean_class1 = np.mean(class1_data, axis=0)
mean_class2 = np.mean(class2_data, axis=0)
mean_class3 = np.mean(class3_data, axis=0)
print(mean_class1)
print(mean_class2)
print("Discriminant functions for 4 dimensions")
bayes_classifier(sigma_class1, sigma_class2, mean_class1, mean_class2, 1)
bayes_classifier(sigma_class2, sigma_class3, mean_class2, mean_class3, 2)
bayes_classifier(sigma_class3, sigma_class1, mean_class3, mean_class1, 2)

df.drop(columns=["SepalLengthCm", "SepalWidthCm"], inplace=True)

class1_data = df.iloc[0:40, :].to_numpy()
class2_data = df.iloc[50:90, :].to_numpy()
class3_data = df.iloc[100:140, :].to_numpy()

sigma_class1 = np.cov(class1_data, rowvar=False)
sigma_class2 = np.cov(class2_data, rowvar=False)
sigma_class3 = np.cov(class3_data, rowvar=False)

print("\nFor 2 dimensions\n")
print("Covariance matrix of class 1")
print(sigma_class1)
print("Covariance matrix of class 2")
print(sigma_class2)
print("Covariance matrix of class 3")
print(sigma_class3)

mean_class1 = np.mean(class1_data, axis=0)
mean_class2 = np.mean(class2_data, axis=0)
mean_class3 = np.mean(class3_data, axis=0)
print("This belongs in Case 3")

bayes_boundary(sigma_class1, sigma_class2, mean_class1, mean_class2, "r", 1)
bayes_boundary(sigma_class2, sigma_class3, mean_class2, mean_class3, "g", 2)
bayes_boundary(sigma_class3, sigma_class1, mean_class3, mean_class1, "b", 3)

x_vals = class1_data[:, 0]
y_vals = class1_data[:, 1]
plt.scatter(x_vals, y_vals, c="cyan", label="Iris-setosa")

x_vals = class2_data[:, 0]
y_vals = class2_data[:, 1]
plt.scatter(x_vals, y_vals, c="orange", label="Iris-versicolor")

x_vals = class3_data[:, 0]
y_vals = class3_data[:, 1]
plt.scatter(x_vals, y_vals, c="purple", label="Iris-virginica")

plt.legend()
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")

plt.show()
