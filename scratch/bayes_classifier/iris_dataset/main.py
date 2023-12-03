import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


df = pd.read_csv('iris.csv')
#print(df.info())

#separate train and test for each class
def train_test(df):
    num_C1 = 40
    num_C2 = 40
    num_C3 = 40
    Train_1 = pd.DataFrame()
    Train_2 = pd.DataFrame()
    Train_3 = pd.DataFrame()
    for ind,row in df.iterrows():
        new_row = pd.DataFrame({'Id' : [row.Id], 'SepalLengthCm' : [row.SepalLengthCm], 'SepalWidthCm': [row.SepalWidthCm], 'PetalLengthCm':[row.PetalLengthCm], 'PetalWidthCm': [row.PetalWidthCm]})
        if row.Species == 'Iris-setosa' and num_C1 > 0:     
            Train_1 = pd.concat([Train_1, new_row], ignore_index = True)
            num_C1 = num_C1 - 1
        elif row.Species == 'Iris-versicolor' and num_C2 > 0:
            Train_2 = pd.concat([Train_2, new_row], ignore_index = True)
            num_C2 = num_C2 - 1
        elif row.Species == 'Iris-virginica' and num_C3 > 0:
            Train_3 = pd.concat([Train_3, new_row], ignore_index = True)
            num_C3 = num_C3 - 1
    
    
    for ind,row in df.iterrows():
        for id,rows in Train_1.iterrows():
            if row.Id == rows.Id:
                df.drop(df[df['Id'] == row.Id].index, inplace = True)
    for ind,row in df.iterrows():
        for id,rows in Train_2.iterrows():
            if row.Id == rows.Id:
                df.drop(df[df['Id'] == row.Id].index, inplace = True)
    for ind,row in df.iterrows():
        for id,rows in Train_3.iterrows():
            if row.Id == rows.Id:
                df.drop(df[df['Id'] == row.Id].index, inplace = True)

    return Train_1, Train_2, Train_3, df

#find apriori probability
def apriori():
    P_omega_1 = 40/120
    P_omega_2 = 40/120
    P_omega_3 = 40/120
    return P_omega_1, P_omega_2, P_omega_3

#finding mean vector
def mean_vector(Train_1, Train_2, Train_3):
    t_1 = Train_1.drop(['Id'], axis=1)
    t_2 = Train_2.drop(['Id'], axis=1)
    t_3 = Train_3.drop(['Id'], axis=1)
    M_1 = np.mean(t_1.T, axis = 1)
    M_2 = np.mean(t_2.T, axis = 1)
    M_3 = np.mean(t_3.T, axis = 1)
    return M_1, M_2, M_3

#finding covariance matrix
def cov_matrix(Train_1, Train_2, Train_3):
    t_1 = Train_1.drop(['Id'], axis=1)
    t_2 = Train_2.drop(['Id'], axis=1)
    t_3 = Train_3.drop(['Id'], axis=1)
    cov_1 = t_1.cov()
    cov_2 = t_2.cov()
    cov_3 = t_3.cov()
    return cov_1, cov_2, cov_3

#likelihood calculation
def multivariate(sample, cov, mean):
    #finding the determinant of cov matrix
    deter = np.linalg.det(cov)
    a = 1 / (pow((2 * math.pi),(2)) * pow(deter,0.5))
    #finding the inverse of cov 
    b = -0.5 * ((sample - mean).T @ np.linalg.inv(cov) @ (sample - mean))
    c = math.exp(b)
    d = a * c
    return d
    
def evidence(P_omega_1, P_omega_2, P_omega_3, P_given_omega_1, P_given_omega_2, P_given_omega_3):
    evi = 0
    evi = P_omega_1*P_given_omega_1 + P_omega_2*P_given_omega_2 + P_omega_3*P_given_omega_3
    return evi

def posterior(sample, Train_1, Train_2, Train_3):
    P_omega_1, P_omega_2, P_omega_3 = apriori()
    mean_vector_1, mean_vector_2, mean_vector_3 = mean_vector(Train_1, Train_2, Train_3)
    cov_1, cov_2, cov_3 = cov_matrix(Train_1, Train_2, Train_3)
    P_given_omega_1 = multivariate(sample, cov_1, mean_vector_1)
    P_given_omega_2 = multivariate(sample, cov_2, mean_vector_2)
    P_given_omega_3 = multivariate(sample, cov_3, mean_vector_3)
    evi = evidence(P_omega_1, P_omega_2, P_omega_3, P_given_omega_1, P_given_omega_2, P_given_omega_3)
    a = (P_given_omega_1 * P_omega_1)/evi
    b = (P_given_omega_2 * P_omega_2)/evi
    c = (P_given_omega_3 * P_omega_3)/evi
    prob = { 'Iris-setosa' : a, 'Iris-versicolor' : b, 'Iris-virginica' : c}
    pred = max(prob, key = lambda x: prob[x])
    #print("Prediction for sample = ", pred)
    return pred

def plotting(y, accuracy, label):
    plt.plot(y, accuracy)
    plt.title(label)
    plt.xlabel("Testcase")
    plt.ylabel("Accuracy")
    plt.show()

def testing():
    Train_1, Train_2, Train_3, Test = train_test(df)
    accuracy = []
    class_accuracy_1 = []
    class_accuracy_2 = []
    class_accuracy_3 = []
    y = []
    for i in range(10):
        y.append(i)
    #print(Test.info())
    for ind, row in Test.iterrows():
        test_case = row.drop(['Id', 'Species'])
        pred = posterior(test_case , Train_1, Train_2, Train_3)
        if row.Species == pred:
            if pred == 'Iris-setosa':
                class_accuracy_1.append(1)
            elif pred == 'Iris-versicolor':
                class_accuracy_2.append(1)
            elif pred == 'Iris-virginica':
                class_accuracy_3.append(1)    
            accuracy.append(1)
        else:
            if pred == 'Iris-setosa':
                class_accuracy_1.append(0)
            elif pred == 'Iris-versicolor':
                class_accuracy_2.append(0)
            elif pred == 'Iris-virginica':
                class_accuracy_3.append(0) 
            accuracy.append(0)
    s = sum(accuracy)
    l = len(accuracy)
    avg = s/l
    s1 = sum(class_accuracy_1)
    l1 = len(class_accuracy_1)
    s2 = sum(class_accuracy_2)
    l2 = len(class_accuracy_2)
    s3 = sum(class_accuracy_3)
    l3 = len(class_accuracy_3)
    avg1 = s1/l1
    avg2 = s2/l2
    avg3 = s3/l3
    print("Class 1 accuracy = ", avg1*100)
    print("Class 2 accuracy = ", avg2*100)
    print("Class 3 accuracy = ", avg3*100)
    print("Total Accuracy = ", avg*100)
    plotting(y, class_accuracy_1, "Class 1 Accuracy")
    plotting(y, class_accuracy_2, "Class 2 Accuracy")
    plotting(y, class_accuracy_3, "Class 3 Accuracy")
    plotting(y, accuracy, "Total Accuracy")
    
testing()
