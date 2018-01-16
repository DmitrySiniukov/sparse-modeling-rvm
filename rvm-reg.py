# Author: Daniel Persson Proos
# TODO Take a second look at the basis functions, something might not be right there.
# TODO Implement the convergenceReached functions with relevant convergence criteria
# TODO Make sure regression algorithm works
# TODO Convert to classification algorithm
import numpy as np
import math

def linearBasisFunction(datapoint):
    return datapoint

def polynomialBasisFunction(degree):
    return lambda x: np.power(x,degree)

def createPolynomialDesignMatrix(degree,dataset):
    designMatrix = np.array([])
    for xi in dataset:
        temp = np.array([])
        for i in range(degree+1):
            temp = np.append(temp,np.power(xi,i))
        designMatrix = np.concatenate((designMatrix,temp.reshape(1,-1)))
        print(designMatrix.shape)

    return designMatrix

# This functions filter out infinite alpha values and the columns of the design
# matrix that correspond to them
def filterOutInf(alphaVec,designMatrix):
    nonInfIdx = np.where(alphaVec != math.inf)
    filteredAlphaVec = alphaVec[nonInfIdx]
    filteredDesignMatrix = designMatrix[:,nonInfIdx]
    return filteredAlphaVec, filteredDesignMatrix

# Calculates new alpha values based on corresponding qi and si values (Bishop eq. 7.101)
def updateAlphaI(qi, si):
    qi, si = getqiAndsi(alphaVec,beta,designMatrix,postCov,t,index)
    return math.pow(si,2)/(math.pow(qi,2)-si)

# Calculates new value for beta according to eq. 7.88 in Bishop
def updateBeta(alphaVec,designMatrix,postMean,postCov,t):
    N = designMatrix.shape[0]
    # Calculate gamma sum
    usdAlphaVec, usdDsgnMtx = filterOutInf(alphaVec,designMatrix)
    gammaSum = 0
    for i in range(len(usdAlphaVec)):
        gammaSum += 1 - usdAlphaVec[i]*usdDsgnMtx[i,i]


    result = np.linalg.norm(t-np.multiply(phi,m))**2
    result /= N - gammaSum

# This method of extracting qi and si uses def 7.102-7.107 in Bishop book
# for convenience
def getqiAndsi(alphaVec, beta, designMatrix,postCov,t,index):
    phiI = designMatrix[:,index]
    alphaI = alphaVec[index]
    usdAlphaVec, usdDsgnMtx = filterOutInf(alphaVec,designMatrix)

    Qi = beta*np.transpose(phiI)*t
    Qi -= math.pow(beta,2)*np.transpose(phiI)*usdDsgnMtx*postCov*np.transpose(usdDsgnMtx)*t

    Si = beta*np.transpose(phiI)*phiI
    Si -= math.pow(beta,2)*np.transpose(phiI)*usdDsgnMtx*postCov*np.transpose(usdDsgnMtx)*phiI

    if alphaI == math.inf:
        qi = Qi
        si = Si
    else:
        qi = alphaI*Qi/(alphaI-Si)
        si = alphaI*Si/(alphaI-Si)

    return qi, si

# Calculates the posterior mean and covariance based on basis functions with
# non-infinite alpha values (Bishop eq. 7.82, 7.83)
def getPosteriorMeanAndCov(alphaVec,beta,designMatrix,t):
    usedAlphaVec, usedDesignMatrix = filterOutInf(alphaVec,designMatrix)
    A = np.diag(usedAlphaVec)
    posteriorCov = A + beta*np.transpose(usedDesignMatrix)*usedDesignMatrix
    posteriorCov = np.linalg.inv(posteriorCov)

    posteriorMean = beta*posteriorCov*np.transpose(usedDesignMatrix)*t

    return posteriorMean, posteriorCov

# Creates polynomial labels for the dataset
def createLabels(x):
    return np.array([xi**2 for xi in x])

# Calculates covariance matrix of the margínal likelihood (Bishop eq. 7.93)
def getC(alphaVec, beta, designMatrix):
    C = np.identity(designMatrix.shape[1])
    nonInfIdx = np.where(alphaVec != math.inf)
    usedAlphaVec = alphaVec[nonInfIdx]
    usedDesignMatrix = designMatrix[:,nonInfIdx]
    for i in range(len(usedAlphaVec)):
        phiI = usedDesignMatrix[:,i]
        C += (1/usedAlphaVec[i])*phiI*np.transpose(phiI)

    return C

# This method of extracting qi and si does not need to use the posterior covariance and
# can therefore be used at step 2 of Bishops Sequential sparse bayesian learning algorithm.
# (Bishop eq. 7.98, 7.99)
def getqiAndsiAlterativeMethod(alphaVec, beta, designMatrix, t, index):
    phiI = designMatrix[:,index]
    alphaI = alphaVec[index]
    C = getC(alphaVec,beta,designMatrix)
    CMinusI = C - (1/alphaI)*phiI*np.transpose(phiI)
    CMinusIInv = np.linalg.inv(CMinusI)
    si = np.transpose(phiI)*CMinusIInv*phiI
    qi = np.transpose(phiI)*CMinusIInv*t
    return si, qi



def convergenceReached(newAlphaVec,oldAlphaVec):
    # TODO Formulate this function
    return false

# Implementation of the Sequential Sparse Bayesian Learning Algorithm in
# Bishop p. 352-353
def main():
    dataset = np.linspace(-2,2,20)
    t = createLabels(dataset)

    beta = 1.
    designMatrix = createPolynomialDesignMatrix(4,dataset)
    alphaVec = np.full(designMatrix.shape[1],math.inf)
    alphaVec[0] = 0

    s0, q0 = getqiAndsiAlterativeMethod(alphaVec,beta,designMatrix,t,0)
    alphaVec[0] = updateAlphaI(s0,q0)
    oldAlphaVec = np.zeros(alphaVec.size)

    for i in range(len(alphaVec)):
        if convergenceReached(alphaVec,oldAlphaVec):
            break

        posteriorMean, posteriorCov = getPosteriorMeanAndCov(alphaVec,beta,designMatrix,t)
        oldAlphaVec = alphaVec

        qi, si = getqiAndsi(alphaVec, beta, designMatrix,posteriorCov,t,i)
        if qi**2 > si:
            alphaVec[i] = updateAlphaI(si,qi)
        else:
            alphaI = math.inf

        beta = updateBeta(alphaVec,designMatrix,posteriorMean,posteriorCov,t)




if __name__ == '__main__':
    main()
