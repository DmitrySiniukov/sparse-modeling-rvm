# Implementation and analysis of RVM method in comparison to standard SVM for solving regression and classification problems

This repository includes the implementation of SVM and RVM for both regression and classification. 

Regarding the SVM both the conventional ε-SVM and v-SVM have been implemented. 

For the RVM, to learn the hyperparameters we have implemented an EM update rule and the direct differentiation rule as it is describe by Tipping's patent in [1]. We also implemented the sparse bayesian learning learning algorithm based on [2].


## Files in repository
* data_gen: this python file contains functions that generate or read the datasets from their respective files in the Data folder and prepare it for use by the machine learning models.

* RVM_Classification: RVM for classification.

* rvm_class: RVM for regression

* svm_clas_class: SVM for classification

* svm_reg_class: SVM for regression





[1]. Tipping, M., Microsoft Corp, 2003. Relevance vector machine. U.S. Patent 6,633,857.

[2]. Tipping, M.E. and Faul, A.C., 2003, January. Fast marginal likelihood maximisation for sparse Bayesian models. In AISTATS.
