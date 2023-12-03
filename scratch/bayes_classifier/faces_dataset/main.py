import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


df = pd.read_csv('face.csv')
print(df.info())

#separate train and test for each class
def train_test(df):
    train_1 = pd.DataFrame()
    train_2 = pd.DataFrame()
    test = pd.DataFrame()

    test = pd.concat([df.iloc[0:5, :], df.iloc[400:405, :]], ignore_index=True)
    train_1 = pd.concat([df.iloc[5:400, :]], ignore_index=True)
    train_2 = pd.concat([df.iloc[405:800, :]], ignore_index=True)
    return train_1, train_2, test

#find apriori probability
def apriori(Train_1, Train_2):
    P_omega_1 = len(Train_1[Train_1['Unnamed: 1'] == "male"]) / (len(Train_1)+len(Train_2))
    P_omega_2 = len(Train_2[Train_2['Unnamed: 1'] == "female"]) / (len(Train_1)+len(Train_2))
    return P_omega_1, P_omega_2

#find mean vector:
def mean_vector(Train_1, Train_2):
    t1 = Train_1
    t2 = Train_2
    t1.pop("Unnamed: 1")
    t2.pop("Unnamed: 1")
    Mean_1 = np.mean(t1.T, axis = 1)
    Mean_2 = np.mean(t2.T, axis = 1)
    return Mean_1, Mean_2

#finding covariance matrix
def cov_matrix(Train_1, Train_2):
    cov_1 = Train_1.cov()
    cov_2 = Train_2.cov()
    return cov_1, cov_2

#likelihood calculation
def multivariate(sample, cov, mean):
    #finding the determinant of cov matrix
    deter = np.linalg.det(cov) + pow(10, -100)
    a = 1 / (pow((2 * math.pi),(2)) * pow(abs(deter),0.5))
    #finding the inverse of cov 
    b = -0.5 * ((sample - mean).T @ np.linalg.inv(cov) @ (sample - mean))
    c = math.exp(b)
    d = a * c
    return d

def evidence(P_omega_1, P_omega_2, P_given_omega_1, P_given_omega_2):
    evi = 0
    evi = P_omega_1*P_given_omega_1 + P_omega_2*P_given_omega_2 
    return evi



def posterior(sample, P_omega_1, P_omega_2, mean_vector_1, mean_vector_2, cov_1, cov_2):
    P_given_omega_1 = multivariate(sample, cov_1, mean_vector_1)
    P_given_omega_2 = multivariate(sample, cov_2, mean_vector_2)
    evi = evidence(P_omega_1, P_omega_2, P_given_omega_1, P_given_omega_2)
    if evi == 0:
        return
    a = (P_given_omega_1 * P_omega_1)/evi
    b = (P_given_omega_2 * P_omega_2)/evi
    prob = { 'male' : a, 'female' : b}
    pred = max(prob, key = lambda x: prob[x])
    #print("Prediction for sample = ", pred)
    return pred

def plotting(accuracy, label):
    y = []
    for i in range(len(accuracy)):
        y.append(i)
    plt.plot(y, accuracy)
    plt.title(label)
    plt.xlabel("Testcase")
    plt.ylabel("Accuracy")
    plt.show()

def testing():
    df.pop("Unnamed: 0")
    
    Train_1, Train_2, Test = train_test(df)
    accuracy = []
    class_accuracy_1 = []
    class_accuracy_2 = []
    y = []
    for i in range(6):
        y.append(i)
    #print(Test.info())
    P_omega_1, P_omega_2 = apriori(Train_1, Train_2)
    mean_vector_1, mean_vector_2 = mean_vector(Train_1, Train_2)
    cov_1, cov_2 = cov_matrix(Train_1, Train_2)
    for ind, row in Test.iterrows():
        truth = row.iloc[-1]
        row.pop("Unnamed: 1")
        pred = posterior(row ,P_omega_1, P_omega_2, mean_vector_1, mean_vector_2, cov_1, cov_2)
        if truth == pred:
            if pred == 'male':
                class_accuracy_1.append(1)
            elif pred == 'female':
                class_accuracy_2.append(1)   
            accuracy.append(1)
        else:
            if pred == 'male':
                class_accuracy_1.append(0)
            elif pred == 'female':
                class_accuracy_2.append(0)
            accuracy.append(0)
    s = sum(accuracy)
    l = len(accuracy)
    avg = s/l
    s1 = sum(class_accuracy_1)
    l1 = len(class_accuracy_1)
    s2 = sum(class_accuracy_2)
    l2 = len(class_accuracy_2)
    avg1 = s1/l1
    avg2 = s2/l2
    print("Class 1 accuracy = ", avg1*100)
    print("Class 2 accuracy = ", avg2*100)
    print("Total Accuracy = ", avg*100)
    plotting(class_accuracy_1, "Class 1 Accuracy")
    plotting(class_accuracy_2, "Class 2 Accuracy")
    plotting(accuracy, "Total Accuracy")

testing()

