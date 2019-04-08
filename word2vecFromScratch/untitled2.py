# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:55:57 2019

@author: ADMIN
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
'''np.random.seed(0)
x1 = np.random.normal(loc=-5.5,scale=0.2,size=(10,2))
x2= np.random.normal(loc=-5,scale = 0.1,size=(10,2))

plt.scatter(x1[:,0],x1[:,1])
plt.scatter(x2[:,0],x2[:,1])
data = np.vstack([x1,x2])
y = np.array([0]*10+[1]*10)
w = np.array([-5,5])
b=0
print(w)
def sigmoid(x):
    return 1/(1+np.exp(-x))
for epoch in range(50000):
    a = sigmoid(data@w+b)
    cost = -y*np.log(a)+(1-y)*np.log(1-a)
    loss = np.mean(cost)
    dw = (a-y)@data/20 
    w=w-0.01*dw
    if epoch%10000==0:
        print(loss)
print(w)
      
class LogisticRegression1:
    def __init__(self, lr=0.01, num_iter=50000, fit_intercept=False, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
                
            if(self.verbose ==True and i % 10000 == 0):
                print(f'loss: {loss} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()

mod=LogisticRegression1()
mod.fit(data,y)  
'''
ar = [1,2,3]
ar.extend([2,3,4])



        