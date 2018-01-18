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

def polynomialKernel(xi,xj):
    degree = 2
    base = np.dot(np.transpose(xi),xj)+1
    return np.power(base,degree)

def linearKernel(xi,xj):
    return np.dot(np.transpose(xi),xj)+1

def radialKernel(xi,xj):
    sigma = 1
    exponent = -np.power(np.linalg.norm(xi-xj),2)/(2*np.power(sigma,2))
    return np.exp(exponent)

def sigmoidKernel(xi,xj):
    k = 1
    delta = 1
    return np.tanh(k*np.dot(np.transpose(xi),xj) - delta)


def createKernel(dataset,kernelFn):
        kernel = []
        for xi in dataset:
            row = []
            for xj in dataset:
                row.append(kernelFn(xi,xj))
            kernel.append(row)
        kernel = np.array(kernel)
        bias = np.ones(dataset.shape[0]).reshape((-1,1))
        return np.concatenate((bias,kernel),axis=1)

def createDesignMatrixKernel(dataset,kernelType):
    kernelFn = linearKernel
    if kernelType == "pol":
        kernelFn = polynomialKernel
    elif kernelType == "rad":
        kernelFn = radialKernel
    elif kernelType == "sig":
        kernelFn = sigmoidKernel

    return createKernel(dataset,kernelFn)

# This functions filter out infinite alpha values and the columns of the design
# matrix that correspond to them
def filterOutInf(alphaVec,designMatrix):
    print("Non-filtered design matrix:")
    print(designMatrix.shape)
    print("Alphas:")
    print(alphaVec)
    nonInfIdx = np.where(alphaVec != math.inf)[0]
    print("non-infinite indicies")
    print(nonInfIdx)
    filteredAlphaVec = alphaVec[nonInfIdx]
    filteredDesignMatrix = designMatrix[:,nonInfIdx]
    print("Filtered design matrix:")
    print(filteredDesignMatrix.shape)
    return filteredAlphaVec, filteredDesignMatrix

# Calculates new alpha values based on corresponding qi and si values (Bishop eq. 7.101)
def updateAlphaI(qi, si):
    print("qi: "+str(qi))
    print("si: "+str(si))
    return math.pow(si,2)/(math.pow(qi,2)-si)

# Calculates new value for beta according to eq. 7.88 in Bishop
def updateBeta(alphaVec,designMatrix,postMean,postCov,t):
    N = designMatrix.shape[0]
    # Calculate gamma sum
    usdAlphaVec, usdDsgnMtx = filterOutInf(alphaVec,designMatrix)
    gammaSum = 0
    for i in range(len(usdAlphaVec)):
        gammaSum += 1 - usdAlphaVec[i]*usdDsgnMtx[i,i]


    result = np.linalg.norm(t-np.dot(usdDsgnMtx,postMean))**2
    result /= N - gammaSum
    return result

# This method of extracting qi and si uses def 7.102-7.107 in Bishop book
# for convenience
def getqiAndsi(alphaVec, beta, designMatrix,postCov,t,index):
    phiI = designMatrix[:,index]
    alphaI = alphaVec[index]
    usdAlphaVec, usdDsgnMtx = filterOutInf(alphaVec,designMatrix)

    temp = np.dot(np.transpose(phiI),usdDsgnMtx)
    temp = np.dot(np.dot(temp,postCov),np.transpose(usdDsgnMtx))

    Qi = beta*np.dot(np.transpose(phiI),t)
    Qi -= math.pow(beta,2)*np.dot(temp,t)

    Si = beta*np.dot(np.transpose(phiI),phiI)
    Si -= math.pow(beta,2)*np.dot(temp,phiI)

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
    print("usedDesignMatrix:")
    print(usedDesignMatrix.shape)
    A = np.diag(usedAlphaVec)
    posteriorCov = A + beta*np.dot(np.transpose(usedDesignMatrix),usedDesignMatrix)
    print("post cov:")
    print(posteriorCov.shape)
    posteriorCov = np.linalg.inv(posteriorCov)

    posteriorMean = beta*posteriorCov*np.transpose(usedDesignMatrix)*t

    return posteriorMean, posteriorCov

# Creates polynomial labels for the dataset
def createLabels(x):
    return np.array([xi**2 for xi in x])

# Calculates covariance matrix of the margínal likelihood (Bishop eq. 7.93)
def getC(alphaVec, beta, designMatrix):
    usdDsgnMtx = designMatrix
    usdAlphaVec = alphaVec
    #usdAlphaVec, usdDsgnMtx = filterOutInf(alphaVec,designMatrix)

    N = usdDsgnMtx.shape[0]
    Binv = np.linalg.inv((1/beta)*np.identity(N))
    Ainv = np.linalg.inv(np.diag(usdAlphaVec))
    C = Binv + np.dot(np.dot(usdDsgnMtx,Ainv),np.transpose(usdDsgnMtx))

    return C

# This method of extracting qi and si does not need to use the posterior covariance and
# can therefore be used at step 2 of Bishops Sequential sparse bayesian learning algorithm.
# (Bishop eq. 7.98, 7.99)
def getqiAndsiAlterativeMethod(alphaVec, beta, designMatrix, t, index):
    phiI = designMatrix[:,index]
    alphaI = alphaVec[index]
    C = getC(alphaVec,beta,designMatrix)
    print("C:")
    print(C)
    CMinusI = C - (1/alphaI)*np.dot(phiI,np.transpose(phiI))
    CMinusIInv = np.linalg.inv(CMinusI)
    si = np.dot(np.dot(np.transpose(phiI),CMinusIInv),phiI)
    qi = np.dot(np.dot(np.transpose(phiI),CMinusIInv),t)
    return si, qi

def calculateMarginalLogLikelihood(alphaVec, beta, designMatrix, t):
    usdAlphaVec, usdDsgnMtx = filterOutInf(alphaVec,designMatrix)
    N = usdDsgnMtx.shape[0]
    Binv = np.linalg.inv((1/beta)*np.identity(N))
    Ainv = np.linalg.inv(np.diag(usdAlphaVec))
    likelihoodCov = Binv + usdDsgnMtx*Ainv*np.transpose(usdDsgnMtx)

    logProb = np.log(likelihoodCov)
    logProb += np.transpose(t)*np.linalg.inv(likelihoodCov)*t
    logProb += N*np.log(2*math.pi)
    logProb *=-1/2
    return logProb

def convergenceReached(newLogProb,oldLogProb):
    print("newLogProb:")
    print(newLogProb.shape)
    print("oldLogProb:")
    print(oldLogProb)
    return newLogProb/oldLogProb < math.pow(10,-6)

def optimizeMarginalLikelihoodParams(dataset, t, kernelType):
    beta = 1.
    designMatrix = createDesignMatrixKernel(dataset,kernelType)
    alphaVec = np.full(designMatrix.shape[1],math.inf)

    s0, q0 = getqiAndsiAlterativeMethod(alphaVec,beta,designMatrix,t,0)
    alphaVec[0] = updateAlphaI(s0,q0)

    oldLogLikelihood = math.pow(10,-6)
    newLogLikelihood = 1.
    while True:
        for i in range(len(alphaVec)):
            if convergenceReached(newLogLikelihood,oldLogLikelihood):
                usdAlphaVec, usdDsgnMtx = filterOutInf(alphaVec,designMatrix)
                return usdAlphaVec, beta, usdDsgnMtx

            posteriorMean, posteriorCov = getPosteriorMeanAndCov(alphaVec,beta,designMatrix,t)
            oldAlphaVec = alphaVec

            qi, si = getqiAndsi(alphaVec, beta, designMatrix,posteriorCov,t,i)
            if qi**2 > si:
                alphaVec[i] = updateAlphaI(si,qi)
            else:
                alphaVec[i] = math.inf

            beta = updateBeta(alphaVec,designMatrix,posteriorMean,posteriorCov,t)

            oldLogLikelihood = newLogLikelihood
            newLogLikelihood = calculateMarginalLogLikelihood(alphaVec, beta, designMatrix, t)

# Implementation of the Sequential Sparse Bayesian Learning Algorithm in
# Bishop p. 352-353
def main():
    dataset = np.linspace(-2,2,20)
    t = createLabels(dataset)

    alphaVec, beta, designMatrix = optimizeMarginalLikelihoodParams(dataset,t,"pol")



    print("Alpha:")
    print(alphaVec)
    print("Beta:")
    print(beta)
    print("Design Matrix:")
    print(designMatrix)





if __name__ == '__main__':
    main()
