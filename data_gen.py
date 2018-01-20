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

eps = 1e-30

def Sinc1(N, x0, xN, sigma):
    x = np.linspace(x0, xN, num=N)
    y = np.sin(x) / (x + eps) + np.random.normal(loc = 0, scale = sigma, size = N)
    return x, y

def genSinc1(norm = True, N_tr = 100):
    eps = 1e-30
    N_tst = 1000
    X_tr, Y_tr = Sinc1(N_tr, -10, 10, 0.1)
    X_ts, Y_ts = Sinc1(N_tst, -10, 10, 0.1)
    if norm == True:
        mean = np.mean(X_tr)
        std = np.std(X_tr)
        X_tr = (X_tr - mean) / (2 * std + eps)
        X_ts = (X_ts - mean) / (2 * std + eps)
    return X_tr, Y_tr, X_ts, Y_ts

def Sinc2(N, x0, xN, n0, nN):
    eps = 1e-30
    x = np.linspace(x0, xN, num=N)
    y = np.sin(x) / (x + eps) + np.random.uniform(low = n0, high = nN, size = N)
    return x, y

def genSinc2(norm = True):
    eps = 1e-30
    N_tr = 100
    N_tst = 1000
    X_tr, Y_tr = Sinc2(N_tr, -10, 10, -0.1, 0.1 )
    X_ts, Y_ts = Sinc2(N_tst, -10, 10, -0.1, 0.1)
    if norm == True:
        mean = np.mean(X_tr)
        std = np.std(X_tr)
        X_tr = (X_tr - mean) / (2 * std + eps)
        X_ts = (X_ts - mean) / (2 * std + eps)
    return X_tr, Y_tr, X_ts, Y_ts


def Friedman1(N, norm = True):
    eps = 1e-30
    x1 = x2 = x3 = x4 = x5 = np.linspace(0, 1, num = N)
    x_rest = np.linspace(0, 1, num = N)
    y = 10*np.sin(np.pi*x1*x2) + 20*(x3 - 0.5)**2 \
        + 10*x4 + 5*x5
    x = np.zeros((N,10))
    x[:,0] = x1
    x[:,1] = x2
    x[:,2] = x3
    x[:,3] = x4
    x[:,4] = x5
    x[:,5] = x[:,6] = x[:,7] = x[:,8] = x[:,9] = x_rest
    sigma = np.sqrt(np.var(y)) / 3
    y = y + np.random.normal(loc = 0, scale = sigma, size = N)
    if norm == True:
        y = y / np.abs(np.max(y) + eps)
    print(np.max(y))
    return x,y

def genFriedman1(norm = True):
    eps = 1e-30
    N_tr = 240
    N_tst = 1000
    X_tr, Y_tr = Friedman1(N_tr)
    X_ts, Y_ts = Friedman1(N_tst)
    if norm == True:
        mean = np.mean(X_tr)
        std = np.std(X_tr)
        X_tr = (X_tr - mean) / (2 * std + eps)
        X_ts = (X_ts - mean) / (2 * std + eps)
    return X_tr, Y_tr, X_ts, Y_ts

def Friedman2(N):
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
    return x,y

def genFriedman2(norm = True):
    eps = 1e-30
    N_tr = 240
    N_tst = 1000
    X_tr, Y_tr = Friedman2(N_tr)
    X_ts, Y_ts = Friedman2(N_tst)
    if norm == True:
        mean = np.mean(X_tr)
        std = np.std(X_tr)
        X_tr = (X_tr - mean) / (2 * std + eps)
        X_ts = (X_ts - mean) / (2 * std + eps)
    return X_tr, Y_tr, X_ts, Y_ts

def Friedman3(N):
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
    return x,y

def genFriedman3(norm = True):
    eps = 1e-30
    N_tr = 240
    N_tst = 1000
    X_tr, Y_tr = Friedman3(N_tr)
    X_ts, Y_ts = Friedman3(N_tst)
    if norm == True:
        mean = np.mean(X_tr)
        std = np.std(X_tr)
        X_tr = (X_tr - mean) / (2 * std + eps)
        X_ts = (X_ts - mean) / (2 * std + eps)
    return X_tr, Y_tr, X_ts, Y_ts

def genBoston(norm = True):
    eps = 1e-30
    boston_housing = skdata.load_boston()
    x = boston_housing.data
    y = boston_housing.target
    N = np.shape(x)[0]
    indices = np.arange(0,N)
    random.shuffle(indices)
    # Following the paper we get 481 values for training
    # and the 25 othre for testing
    X_tr = x[0:481]
    Y_tr = y[0:481]
    X_ts = x[481:N]
    Y_ts = y[481:N]
    #X_tr = x[0:10]
    #Y_tr = y[0:10]
    #X_ts = x[300:N]
    #Y_ts = y[300:N]
    if norm == True:
        mean = np.mean(X_tr)
        std = np.std(X_tr)
        X_tr = (X_tr - mean) / (2 * std + eps)
        X_ts = (X_ts - mean) / (2 * std + eps)
    return X_tr, Y_tr, X_ts, Y_ts


def genBanana():
    file = dirData + 'BANANA'
    data = np.loadtxt(file)
    # Seperate to input and output
    train_data = data[0:400, :]
    test_data = data[400:1000, :]
    X_tr = train_data[:, 0:2]
    Y_tr = train_data[:,2] 
    X_ts = test_data[:, 0:2]
    Y_ts = test_data[:,2] 
    return X_tr, Y_tr, X_ts, Y_ts
    
def genWaveform():
    file = dirData + 'WAVEFORM'
    data = np.loadtxt(file)
    # Seperate to input and output
    train_data = data[0:400, :]
    test_data = data[400:1000, :]
    X_tr = train_data[:, 0:21]
    Y_tr = train_data[:,21] 
    X_ts = test_data[:, 0:21]
    Y_ts = test_data[:,21] 
    return X_tr, Y_tr, X_ts, Y_ts

def genDiabetes():
    file1 = dirData + 'PimaDiabetesTrain'
    file2 = dirData + 'PimaDiabetesTest'
    train_data = np.loadtxt(file1)
    test_data = np.loadtxt(file2)
    # Seperate to input and output
    X_tr = train_data[:, 0:7]
    Y_tr = train_data[:,7] 
    X_ts = test_data[:, 0:7]
    Y_ts = test_data[:,7] 
    return X_tr, Y_tr, X_ts, Y_ts
      
        
def genRiplaySynthetic():
    file1 = dirData + 'RiplaySynthetic250'
    file2 = dirData + 'RiplaySynthetic1000'
    train_data = np.loadtxt(file1)
    test_data = np.loadtxt(file2)
    # Seperate to input and output
    X_tr = train_data[:, 0:2]
    Y_tr = train_data[:,2]
    X_ts = test_data[:, 0:2]
    Y_ts = test_data[:,2] 
    return X_tr, Y_tr, X_ts, Y_ts


def genData(dataset = "Sinc1", norm = False, N_tr = 100):
    if dataset == "Sinc1":
        return genSinc1(norm = norm, N_tr = N_tr)
    elif dataset == "Sinc2":
        return genSinc2(norm = norm)
    elif dataset == "F1":
        return genFriedman1(norm = norm)
    elif dataset == "F2":
        return genFriedman2(norm = norm)
    elif dataset == "F3":
        return genFriedman3(norm = norm)
    elif dataset == "Boston":
        return genBoston(norm = norm)
    elif dataset == "Banana":
        return genBanana()
    elif dataset == "Diabetes":
        return genDiabetes()
    elif dataset == "RiplaySynthetic":
        return genRiplaySynthetic()
    elif dataset == "WAVEFORM":
        return genWaveform()
    else:
        print("Possible datasets: \n" + \
                  + "\t Sinc1" + \
                  + "\t Sinc2" + \
                  + "\t F1" + \
                  + "\t F2" + \
                  + "\t F3" + \
                  + "\t Boston" + \
                  + "\t Diabetes" + \
                  + "\t WAVEFORM" + \
                  + "\t Banana")
        return -1
