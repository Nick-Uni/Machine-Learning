# --------------------------------------------------------------------------
# ----------------  System Analysis and Decision Making --------------------
# --------------------------------------------------------------------------
#  Assignment 1: k-NN and Naive Bayes
#  Authors: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division
import numpy as np
import scipy.sparse as sp
import scipy.spatial.distance as p
from scipy.spatial.distance import cdist
number_of_classes = 4

def hamming_distance(X, X_train):
    """
    :param X: set of objects that are going to be compared N1xD
    :param X_train: set of objects compared against param X N2xD
    Functions calculates Hamming distances between all objects from set X  and all object from set X_train.
    Resulting distances are returned as matrices.
    :return: Distance matrix between objects X and X_train X i X_train N1xN2
    """
    X = sp.spmatrix.toarray(X)
    X_train = sp.spmatrix.toarray(X_train)
    return p.cdist(X, X_train, "hamming") * X.shape[1]

def sort_train_labels_knn(Dist, y):
    """
    Function sorts labels of training data y accordingly to probabilities stored in matrix Dist.
    Function returns matrix N1xN2. In each row there are sorted data labels y accordingly to corresponding row of matrix Dist.
    :param Dist: Distance matrix between objests X and X_train N1xN2
    :param y: N2-element vector of labels
    :return: Matrix of sorted class labels ( use metgesort algorithm)
    """

    return y[Dist.argsort(kind='mergesort', axis=1)]


def p_y_x_knn(y, k):
    """
    Function calculates conditional probability p(y|x) for
    all classes and all objects from test set using KNN classifier
    :param y: matrix of sorted labels for training set N1xN2
    :param k: number of nearest neighbours
    :return: matrix of probabilities for objects X
    """
    #number_of_classes = np.unique(y[0, :]).shape[0]


    y = y[:, :k]
    ret = np.zeros(shape=(y.shape[0], number_of_classes), dtype=float)

    for j in range(ret.shape[0]):
        for i in range(ret.shape[1]):
            ret[j, i] = np.count_nonzero((y[j, :]) == (i + 1))

    return ret/k

def classification_error(p_y_x, y_true):
    """
    Function calculates classification error
    :param p_y_x: matrix of predicted probabilities
    :param y_true: set of ground truth labels 1xN.
    Each row of matrix represents distribution p(y|x)
    :return: classification error
    """
    return np.mean((np.argsort(p_y_x, axis=1)[:, -1] + 1) != y_true)

def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: validation data N1xD
    :param Xtrain: training data N2xD
    :param yval: class labels for validation data 1xN1
    :param ytrain: class labels for training data 1xN2
    :param k_values: values of parameter k that are going to be evaluated
    :return: function makes model selection with knn and results tuple best_error,best_k,errors), where best_error is the lowest
    error, best_k - value of k parameter that corresponds to the lowest error, errors - list of error values for
    subsequent values of k (elements of k_values)
    """
    distance = hamming_distance(Xval, Xtrain)
    y_sorted = sort_train_labels_knn(distance, ytrain)
    errors = []
    for k in k_values:
        error_k = classification_error(p_y_x_knn(y_sorted, k), yval)
        errors.append(error_k)
    best_error = min(errors)
    best_k = k_values[errors.index(best_error)]
    return best_error, best_k, errors

def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: labels for training data 1xN
    :return: function calculates distribution a priori p(y) and returns p_y - vector of a priori probabilities 1xM
    """
    N = ytrain.shape[0]
    res = np.zeros(shape=(number_of_classes))
    for i in range(1, 5):
        res[i - 1] = np.count_nonzero(ytrain == i) / N
    return res

def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: training data NxD
    :param ytrain: class labels for training data 1xN
    :param a: parameter a of Beta distribution
    :param b: parameter b of Beta distribution
    :return: Function calculated probality p(x|y) assuming that x takes binary values and elements
    x are independent from each other. Function returns matrix p_x_y that has size MxD.
    """
    Xtrain = Xtrain.A
    N, D = Xtrain.shape
    p_x_y = np.empty((number_of_classes, D))
    ones = np.ones(N)

    for k in range(number_of_classes):
        for i in range(D):
            # Find idicies of examples of class k
            idxes = np.nonzero(ytrain == k + 1)[0]
            # Count how many times each word appears in documents of class k
            word_count = np.count_nonzero(Xtrain[idxes, i] == 1, axis=0)
            # Use the formula provided in the assignment
            p_x_y[k, i] = (word_count + a - 1) / (idxes.shape[0] + a + b - 2)

    return p_x_y


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: vector of a priori probabilities 1xM
    :param p_x_1_y: probability distribution p(x=1|y) - matrix MxD
    :param X: data for probability estimation, matrix NxD
    :return: function calculated probability distribution p(y|x) for each class with the use of Naive Bayes classifier.
     Function returns matrixx p_y_x of size NxM.
    """
    N = np.shape(X)[0]
    M = np.shape(p_y)[0]
    X = X.toarray()

    def f(n, m):
        return np.prod(np.negative(X[n, :]) - p_x_1_y[m, :])
    # Defining vectorized function which outputs numpy array
    g = np.vectorize(f)
    # Constructing an array by executing a function over each coordinate
    result = np.fromfunction(g, shape=(N, M), dtype=int) * p_y
    result /= result @ np.ones(shape=(4, 1))

    return result



def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: training setN2xD
    :param Xval: validation setN1xD
    :param ytrain: class labels for training data 1xN2
    :param yval: class labels for validation data 1xN1
    :param a_values: list of parameters a (Beta distribution)
    :param b_values: list of parameters b (Beta distribution)
    :return: Function makes a model selection for Naive Bayes - that is selects the best values of a i b parameters.
    Function returns tuple (error_best, best_a, best_b, errors) where best_error is the lowest error,
    best_a - a parameter that corresponds to the lowest error, best_b - b parameter that corresponds to the lowest error,
    errors - matrix of errors for each pair (a,b)
    """
    error_best = 10
    A = len(a_values)
    B = len(b_values)
    errors = np.zeros(shape=(A, B), dtype=np.float64)
    # Remove single-dimensional entries from the shape of an array.
    p_y = np.squeeze(np.asarray(estimate_a_priori_nb(ytrain)))
    best_a_index = 0
    best_b_index = 0
    for a in range(A):
        for b in range(B):
            p_x_1_y = estimate_p_x_y_nb(Xtrain, ytrain, a_values[a], b_values[b])
            p_y_x = p_y_x_nb(p_y, p_x_1_y, Xval)
            error = classification_error(p_y_x, yval)
            errors[a, b] = error
            if error < error_best:
                error_best = error
                best_a_index = a
                best_b_index = b
    return error_best, a_values[best_a_index], b_values[best_b_index], errors