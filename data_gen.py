import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn.datasets as skdata
import scipy.optimize as optimize
import math
import scipy.stats as sts
import time
from rvm_class import RVM_reg
from svm_reg_class import SVM_reg
from collections import deque
from numpy import linalg
dirData = 'Data/'

def Sinc1(N, x0, xN, sigma, norm = True):
    eps = 1e-30
    x = np.linspace(x0, xN, num=N)
    y = np.sin(x) / (x + eps) + np.random.normal(loc = 0, scale = sigma, size = N)
    if norm == True:
        y = y / np.abs(np.max(y) + eps)
    return x, y

def genSinc1(norm = True):
    N_tr = 100
    N_tst = 1000
    X_tr, Y_tr = Sinc2(N_tr, -10, 10, 0.1, norm)
    X_ts, Y_ts = Sinc2(N_tst, -10, 10, 0.1, norm)
    return X_tr, Y_tr, X_ts, Y_ts

def Sinc2(N, x0, xN, n0, nN, norm = True):
    eps = 1e-30
    x = np.linspace(x0, xN, num=N)
    y = np.sin(x) / (x + eps) + np.random.uniform(low = n0, high = nN, size = N)
    if norm == True:
        y = y / np.abs(np.max(y) + eps)
    return x, y

def genSinc2(norm = True):
    N_tr = 100
    N_tst = 1000
    X_tr, Y_tr = Sinc2(N_tr, -10, 10, -0.1, 0.1, norm )
    X_ts, Y_ts = Sinc2(N_tst, -10, 10, -0.1, 0.1, norm)
    return X_tr, Y_tr, X_ts, Y_ts

def Friedman1(N, norm = True):
    eps = 1e-30
    x1 = x2 = x3 = x4 = x5 = np.linspace(0, 1, num = N)
    y = 10*np.sin(np.pi*x1*x2) + 20*(x3 - 0.5)**2 \
        + 10*x4 + 5*x5
    x = np.zeros((N,5))
    x[:,0] = x1
    x[:,1] = x2
    x[:,2] = x3
    x[:,3] = x4
    x[:,4] = x5
    sigma = np.sqrt(np.var(y)) / 3
    y = y + np.random.normal(loc = 0, scale = sigma, size = N)
    if norm == True:
        y = y / np.abs(np.max(y) + eps)
    return x,y

def genFriedman1(norm = True):
    N_tr = 240
    N_tst = 1000
    X_tr, Y_tr = Friedman1(N_tr, norm)
    X_ts, Y_ts = Friedman1(N_tst, norm)
    return X_tr, Y_tr, X_ts, Y_ts

def Friedman2(N, norm = True):
    eps = 1e-30
    x1 = np.linspace(0, 100, num = N)
    x2 = np.linspace(40*np.pi, 560*np.pi, num = N)
    x3 = np.linspace(0, 1, num = N)
    x4 = np.linspace(1, 11, num = N)
    y_squared = x1**2 + (x2*x3 - 1/(x2*x4 + eps))**2 + eps
    y = np.sqrt(y_squared)
    x = np.zeros((N,4))
    x[:,0] = x1
    x[:,1] = x2
    x[:,2] = x3
    x[:,3] = x4
    sigma = np.sqrt(np.var(y)) / 3
    y = y + np.random.normal(loc = 0, scale = sigma, size = N)
    if norm == True:
        y = y / np.abs(np.max(y) + eps)
    return x,y

def genFriedman2(norm = True):
    N_tr = 240
    N_tst = 1000
    X_tr, Y_tr = Friedman2(N_tr, norm)
    X_ts, Y_ts = Friedman2(N_tst, norm)
    return X_tr, Y_tr, X_ts, Y_ts

def Friedman3(N, norm):
    eps = 1e-30
    x1 = np.linspace(0, 100, num = N)
    x2 = np.linspace(40*np.pi, 560*np.pi, num = N)
    x3 = np.linspace(0, 1, num = N)
    x4 = np.linspace(1, 11, num = N)
    y_tan = (x2 * x3 - 1 / (x2 * x4 + eps)) / (x1 + eps)
    y = np.tanh(y_tan)
    x = np.zeros((N,4))
    x[:,0] = x1
    x[:,1] = x2
    x[:,2] = x3
    x[:,3] = x4
    sigma = np.sqrt(np.var(y)) / 3
    y = y + np.random.normal(loc = 0, scale = sigma, size = N)
    if norm == True:
        y = y / np.abs(np.max(y) + eps)
    return x,y

def genFriedman3(norm = True):
    N_tr = 240
    N_tst = 1000
    X_tr, Y_tr = Friedman3(N_tr, norm)
    X_ts, Y_ts = Friedman3(N_tst, norm)
    return X_tr, Y_tr, X_ts, Y_ts

def genBoston(norm = True):
    eps = 1e-30
    boston_housing = skdata.load_boston()
    x = boston_housing.data
    y = boston_housing.target
    if norm == True:
        y = y / np.abs(np.max(y) + eps)
    N = np.shape(x)[0]
    indices = np.arange(0,N)
    random.shuffle(indices)
    # Following the paper we get 481 values for training
    # and the 25 othre for testing
    X_tr = x[0:481]
    Y_tr = y[0:481]
    X_ts = x[481:N]
    Y_tx = x[481:N]
    #X_tr = x[0:10]
    #Y_tr = y[0:10]
    #X_ts = x[300:N]
    #Y_ts = y[300:N]
    return X_tr, Y_tr, X_ts, Y_ts

def genData(dataset = "Sinc1"):
    if dataset == "Sinc1":
        return genSinc1()
    elif dataset == "Sinc2":
        return genSinc2()
    elif dataset == "F1":
        return genFriedman1()
    elif dataset == "F2":
        return genFriedman2()
    elif dataset == "F3":
        return genFriedman3()
    elif dataset == "Boston":
        return genBoston()
    else:
        print("Possible datasets: \n" + \
                  + "\t Sinc1" + \
                  + "\t Sinc2" + \
                  + "\t F1" + \
                  + "\t F2" + \
                  + "\t F3" + \
                  + "\t Boston")
        return -1