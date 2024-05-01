# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 18:50:06 2022

@author: huber
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

def model(X, Y, learning_rate, iteration):
    #size of our y
    m = Y.size
    #0 os matrix
    theta = np.zeros((X.shape[1], 1))
    costsarr = []
    
    for i in range(iteration):
        y_prediction = np.dot(X, theta)
        cost = (1/(2*m))*np.sum(np.square(y_prediction - Y))
        d_theta = (1/m)*np.dot(X.T, y_prediction - Y)
        theta = theta - learning_rate*d_theta
        costsarr.append(cost)
        # to print the cost for 10 times
        if(i%(iteration/10) == 0):
            print("Cost is :", cost)
            
    return theta, costsarr


def linearregression(df):

    train, test = train_test_split(df, test_size=0.2)
    
    train_data = train.values
    Y = train_data[:, -1].reshape(train_data.shape[0], 1)
    X = train_data[:, :-1]
    
    test_data = test.values
    Y_test = test_data[:, -1].reshape(test_data.shape[0], 1)
    X_test = test_data[:, :-1]
    
    X = np.vstack((np.ones((X.shape[0], )), X.T)).T
    X_test = np.vstack((np.ones((X_test.shape[0], )), X_test.T)).T
    #here we declare number of iterations
    iteration = 2000
    #here we declare number of learning rate
    learning_rate = 0.005
    theta, costsarr = model(X, Y, learning_rate = learning_rate, iteration = iteration)
    
    costsarr = costsarr[:2000]
    givenrange = np.arange(0, 2000)
    plt.plot(givenrange, costsarr)
    #z jakiegos powodu nie mogłem wstawic plt.xlabel i plt.ylabel
    plt.title('koszt/iteracja')
    plt.show()
    y_prediction = np.dot(X_test, theta)
    err = (1/X_test.shape[0])*np.sum(np.abs(y_prediction - Y_test))
    
    print("error is :", err*100, "%")
    print("Accuracy is :", (1- err)*100, "%")


#i converted data .csv to .xlsx 
#data from .csv was written in columns not with commas
#i give data from .xlsx to dataframe
 
df = pd.read_excel('data.xlsx')
linearregression(df)