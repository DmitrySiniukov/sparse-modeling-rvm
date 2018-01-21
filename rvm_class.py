# Import of the packages
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import math
import scipy.stats as sts
from collections import deque
from numpy import linalg

__author__ = "Borja Rodriguez Galvez and Matthaios Stylianidis"

class ParameterValueError(Exception):
    """ Custom exception raised when a parameter has an invalid value."""
    pass

# RVM class
class RVM_reg:
    """ Relevance Vector Machine (RVM)
    
    Implementation of RVM for regression.
    
    Attributes:
        kerType: A string of the type of the desired kernel.
        rvmType: A string denoting the RVM type to be used. The string "EM" denotes
            an EM-like algorithm will be used to estimate the hyperparameters, the
            string "DD" denotes the direct differentiation approach while "SSBL"
            denotes sequential sparse bayesian learning.
        p: Integer value denoting the degree of the polynomial kernel.
        sigma: Float value denoting the smoothing factor of the Gaussian kernel.
        kappa: Float value denoting the scaling parameter of the sigmoid kernel.
        delta: Float value denoting the translation parameter of the sigmoid kernel.
        bTrained: boolean value which becomes true once the RVM has been trained.
    """

    EPSILON_CONV = 1e-6
    EPSILON_UF = 1e-30
    TH_RV = 1e5
    INFINITY = 1e20
    maxEpochs = 5000
    
    def __init__(self, kerType = 'poly', rvmType = "EM", p = 1, sigma = 1, 
                 kappa = 1, delta = 1):
        """ Initializes the RVM class (constructor).
        
            Raises:
                ParameterValueError: An error occured because a parameter had an
                    invalid value.  
        """
        # Check if the kernel type chosen is valid
        kerTypes = ['linear', 'poly', 'radial', 'sigmoid']
        if kerType not in kerTypes:
            raise ParameterValueError("ParameterValueError: The string " + kerType +                                        " does not denote a valid kernel type")
        # Check if the string denoting the rvmType has a valid value
        if rvmType != 'EM' and rvmType != 'DD' and rvmType != "SSBL":
            raise ParameterValueError('ParameterValueError: ' + rvmType,                                        " is not a valid RVM type value. Enter 'EM', 'DD' or 'SSBL' as a value. ")
       
        self.kerType = kerType
        self.rvmType = rvmType
        self.p = p
        self.sigma = sigma
        self.kappa = kappa
        self.delta = delta
        self.bTrained = False

    def kernel(self, x, y):
        """ Kernel computation.
        
        It computes the kernel value based on the dot product
        between two vectors.
        
        Args:
            x: Input vector.
            y: Other input vector.
            
        Returns:
            The computed kernel value.
        """  
        if self.kerType == "linear":
            k = np.dot(x,y) + 1
        elif self.kerType == "poly":
            k = (np.dot(x,y) + 1) ** self.p
        elif self.kerType == "radial":
            k = math.exp(-(np.dot(x-y,x-y))/(2*self.sigma))
        elif self.kerType == "sigmoid":
            k = math.atanh(self.kappa * np.dot(x,y) - self.delta)

        return k
    
    def getKernelMatrix(self, X, training = True):
        """ Evaluates the kernel matrix K given a set of input samples (training).

        Args:
            X_tr: A NxM matrix with a M dimensional training input sample
                in each row.

        Returns:
            An NxN Kernel matrix where N is the number of input samples.
        """
        N = X.shape[0]
        if training == True:
            K = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    K[i,j] = self.kernel(X[i], X[j])
        else:
            N_sv = self.X_sv.shape[0]
            K = np.zeros((N,N_sv))
            for i in range(N):
                for j in range(N_sv):
                    K[i,j] = self.kernel(X[i], self.X_sv[j])
        return K

    def getGammaValues(self, alpha_values, Sigma):
        """Evaluates the gamma values.
        
        Args:
            alpha_values: N-dimensional vector with the hyperparameters of
                the marginal likelihood.
            Sigma: NxN covariance matrix of the posterior

        Returns: A N-dimensional vector with the gamma values where 
            gamma_values[i] = 1 - alpha_values[i] * Sigma[i][i]
        """
        N = alpha_values.shape[0]
        gamma_values = 1 - np.multiply(alpha_values, np.diag(Sigma))
        return gamma_values
        
    def getAlphaValues(self, Sigma, mu, gamma_values):
        """Evaluates the alpha values.

        Args:
            Sigma: NxN covariance matrix of the posterior
            mu: mean of the posterior
            gamma_values: N-dimensional vector with gamma_values

        Returns: A N-dimensional vector with the alpha_values
        """        
        N = Sigma.shape[0]
        alpha_values = np.zeros(N)
        if self.rvmType == "EM":
            cond_low = (np.diag(Sigma) + mu**2) < self.EPSILON_UF
            cond_high = (np.diag(Sigma) + mu**2) > self.INFINITY
            ncond = np.logical_and(np.logical_not(cond_low), np.logical_not(cond_high))
            alpha_values[cond_low] = self.INFINITY
            alpha_values[cond_high] = 0
            alpha_values[ncond] = 1 / (np.diag(Sigma)[ncond] + mu[ncond]**2)
        elif self.rvmType == "DD":
            cond_low = (mu**2) < self.EPSILON_UF
            cond_high = (mu**2) > self.INFINITY
            ncond = np.logical_and(np.logical_not(cond_low), np.logical_not(cond_high))
            alpha_values[cond_low] = self.INFINITY
            alpha_values[cond_high] = 0
            alpha_values[ncond] = gamma_values[ncond] / (mu[ncond]**2)
        return alpha_values

    
    def train(self, X_tr, Y_tr):
        """ RVM training method
        
        Applies an EM-like algorithm or direct differentiation to estimate the
        optimal hyperparameters (alpha and sigma) needed to make predictions
        using the marginal likelihood.
        Alternatively, applies the Sequential Sparse Bayesian Algorithm to estimate
        the optimal hyperparemeters (alpha and sigma) more efficiently.
        
        
        Args:
            X_tr: A matrix with a training input sample in each row.
            Y_tr: A vector with the output values of each input sample
                in X_tr.
        
        Returns:
            None
        """
        # Get number of training data samples
        N = X_tr.shape[0]
        # Initialize the sigma squared value and the B matrix
        sigma_squared = np.var(Y_tr) * 0.1
        B = np.identity(N) / sigma_squared
        # Calculate kernel matrix K and append a column with ones in the front 
        K = self.getKernelMatrix(X_tr)        
        K = np.hstack((np.ones(N).reshape((N, 1)), K))
        

        if(self.rvmType == "EM" or self.rvmType =="DD"):
            ''' 
            Implementation based on the following paper Tipping, Michael. 
            "Relevance vector machine." U.S. Patent No. 6,633,857. 14 Oct. 2003.
            '''
            # Initialize the alpha values (weight precision values) and the A matrix
            alpha_values = np.ones(N + 1)
            A = np.diag(alpha_values)
            # Calculate Sigma and mu based on the initialized parameters
            try:
                Sigma = np.linalg.inv(K.T.dot(B).dot(K) + A)
            except linalg.LinAlgError:
                Sigma = np.linalg.pinv(K.T.dot(B).dot(K) + A)
            mu = Sigma.dot(K.T).dot(B).dot(Y_tr)
            # Calculate initial gamma values
            gamma_values = self.getGammaValues(alpha_values, Sigma)

            # Approximate optimal hyperparameter values iteratively
            for epoch in range(self.maxEpochs):
                # Evaluate alpha values
                next_alpha_values = self.getAlphaValues(Sigma, mu, gamma_values)
                # Evaluate sigma value
                next_sigma_squared = (np.linalg.norm(Y_tr - K.dot(mu)) ** 2) / (N - np.sum(gamma_values))
                # Check if algorithm has converged (variation of alpha and sigma)
                if (np.sum(np.absolute(next_alpha_values - alpha_values)) < self.EPSILON_CONV and
                    abs(next_sigma_squared - sigma_squared) < self.EPSILON_CONV):
                        break
                # If algorithm has not converged, update all the variables
                alpha_values = next_alpha_values
                sigma_squared = next_sigma_squared
                A = np.diag(alpha_values)
                B = np.identity(N) / sigma_squared
                try:
                    Sigma = np.linalg.inv(K.T.dot(B).dot(K) + A)
                except linalg.LinAlgError:
                    Sigma = np.linalg.pinv(K.T.dot(B).dot(K) + A) 
                mu = Sigma.dot(K.T).dot(B).dot(Y_tr)
                gamma_values = self.getGammaValues(alpha_values, Sigma)
                
            # We store the support vectors and other important variables
            cond_sv = alpha_values < self.TH_RV
            self.X_sv = X_tr[cond_sv[1:N+1]]
            self.Y_sv = Y_tr[cond_sv[1:N+1]]
            self.mu = mu[cond_sv]
            self.Sigma = Sigma[cond_sv][:,cond_sv]
            self.sigma_squared = sigma_squared
            
        elif self.rvmType == "SSBL":     
            """
            Implementation based on the following paper: Tipping, M.E. and Faul, A.C., 2003, January.
            Fast marginal likelihood maximisation for sparse Bayesian models. In AISTATS.
            """
            
            # 2. Initialize one alpha value and set all the others to infinity.
            alpha_values = np.zeros(N + 1) + self.INFINITY
            basis_column = K[:,0]
            phi_norm = np.linalg.norm(basis_column)
            alpha_values[0] = (phi_norm **2) / ((np.linalg.norm(basis_column.dot(Y_tr)) ** 2)                                           / (phi_norm ** 2) - sigma_squared)
            included_cond = np.zeros(N + 1, dtype=bool)
            included_cond[0] = True
            
            # 3. Initialize Sigma and mu
            A = np.zeros(1) + alpha_values[0]
            basis_column = basis_column.reshape((N, 1)) # Reshape so that it can be transposed
            Sigma = 1 / (basis_column.T.dot(B).dot(basis_column) + A)
            mu = Sigma.dot(basis_column.T).dot(B).dot(Y_tr)

            # 3. Initialize q and s for all bases
            q = np.zeros(N + 1)
            Q = np.zeros(N + 1)
            s = np.zeros(N + 1)
            S = np.zeros(N + 1)
            Phi = basis_column
            for i in range(N + 1):
                basis = K[:, i]
                tmp_1 = basis.T.dot(B)
                tmp_2 = tmp_1.dot(Phi).dot(Sigma).dot(Phi.T).dot(B)
                Q[i] = tmp_1.dot(Y_tr) - tmp_2.dot(Y_tr)
                S[i] = tmp_1.dot(basis) - tmp_2.dot(basis)
            denom = (alpha_values - S)
            s = (alpha_values * S) / denom
            q = (alpha_values * Q) / denom
            
            # Create queue with indices to select candidates for update
            queue = deque([i for i in range(N + 1)])
            # Start updating the model iteratively
            for epoch in range(self.maxEpochs):
                # 4. Pick a candidate basis vector from the start of the queue and put it at the end
                basis_idx = queue.popleft()
                queue.append(basis_idx)
                
                # 5. Compute theta
                theta = q ** 2 - s
                
                next_alpha_values = np.copy(alpha_values)
                next_included_cond = np.copy(included_cond)
                if theta[basis_idx] > 0 and alpha_values[basis_idx] < self.INFINITY:
                    # 6. Re-estimate alpha
                    next_alpha_values[basis_idx] = s[basis_idx] ** 2 / (q[basis_idx] ** 2 - s[basis_idx])
                    pass
                elif theta[basis_idx] > 0 and alpha_values[basis_idx] >= self.INFINITY:
                    # 7. Add basis function to the model with updated alpha
                    next_alpha_values[basis_idx] = s[basis_idx] ** 2 / (q[basis_idx] ** 2 - s[basis_idx])
                    next_included_cond[basis_idx] = True
                elif theta[basis_idx] <= 0 and alpha_values[basis_idx] < self.INFINITY:
                    # 8. Delete theta basis function from model and set alpha to infinity
                    next_alpha_values[basis_idx] = self.INFINITY
                    next_included_cond[basis_idx] = False
                    
                # 9. Estimate noise level
                gamma_values = 1 - np.multiply(alpha_values[included_cond], np.diag(Sigma))
                next_sigma_squared = (np.linalg.norm(Y_tr - Phi.dot(mu)) ** 2) / (N - np.sum(gamma_values))
                
                # 11. Check for convergence
                # Check if algorithm has converged (variation of alpha and sigma)
                not_included_cond = np.logical_not(included_cond)
                if (np.sum(np.absolute(next_alpha_values[included_cond] - alpha_values[included_cond]))                             < self.EPSILON_CONV) and all(th <= 0 for th in theta[not_included_cond]):              
                        break
                
                # 10. Recompute/update  Sigma and mu as well as s and q
                alpha_values = next_alpha_values
                sigma_squared = next_sigma_squared
                included_cond = next_included_cond
                A = np.diag(alpha_values[included_cond])
                B = np.identity(N) / sigma_squared
                Phi = K[:, included_cond]
                # Compute Sigma
                tmp = Phi.T.dot(B).dot(Phi) + A
                if(tmp.shape[0] == 1):
                    Sigma = 1 / tmp
                else:
                    try:
                        Sigma = np.linalg.inv(tmp)
                    except linalg.LinAlgError:
                        Sigma = np.linalg.pinv(tmp)
                    
                # Compute mu
                mu = Sigma.dot(Phi.T).dot(B).dot(Y_tr)
                # Update s and q
                for i in range(N + 1):
                    basis = K[:, i]
                    tmp_1 = basis.T.dot(B)
                    tmp_2 = tmp_1.dot(Phi).dot(Sigma).dot(Phi.T).dot(B)
                    Q[i] = tmp_1.dot(Y_tr) - tmp_2.dot(Y_tr)
                    S[i] = tmp_1.dot(basis) - tmp_2.dot(basis)
                denom = (alpha_values - S)
                s = (alpha_values * S) / denom
                q = (alpha_values * Q) / denom
            ##print(epoch)
            # We store the relevance vectors and other important variables
            alpha_values = alpha_values[included_cond]
            X_tr = X_tr[included_cond[1:N+1]]
            Y_tr = Y_tr[included_cond[1:N+1]]
            
            
            cond_sv = alpha_values < self.TH_RV
            if alpha_values.shape[0] != X_tr.shape[0]:
                self.X_sv = X_tr[cond_sv[1:N+1]]
                self.Y_sv = Y_tr[cond_sv[1:N+1]]
            else:
                self.X_sv = X_tr[cond_sv]
                self.Y_sv = Y_tr[cond_sv]
                
            self.mu = mu[cond_sv]
            self.Sigma = Sigma[cond_sv][:,cond_sv]
            self.sigma_squared = sigma_squared
                   
        self.bTrained = True
    
    def pred(self, X):
        """Predicts the classes for a number of input data
        
        Args:
            X: matrix with input data where each row represents a sample.
            
        Returns:
            y: A tuple with vector with the predicted class for each input sample
                and the error variance.
            
        Raises:
            UntrainedModelError: Error that occurs when this function is called
                before calling the 'train' function.
        """
        
        if self.bTrained == False:
            raise UntrainedModelError("UntrainedModelError: The SVM model has not been trained.")
        
        N = X.shape[0]
        K = self.getKernelMatrix(X, training = False)  
        N_sv = np.shape(self.X_sv)[0]
        if np.shape(self.mu)[0] != N_sv:
            K = np.hstack((np.ones(N).reshape((N, 1)), K))
        y = K.dot(self.mu)
        err_var = self.sigma_squared + K.dot(self.Sigma).dot(K.T)
        return y, np.sqrt(np.diag(err_var))

    def getSV(self):
        return self.X_sv, self.Y_sv
               

