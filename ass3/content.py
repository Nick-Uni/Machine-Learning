# --------------------------------------------------------------------------
# ----------------  System Analysis and Decision Making --------------------
# --------------------------------------------------------------------------
#  Assignment 1: Logistic regression
#  Authors: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------


import numpy as np

def sigmoid(x):
    '''
    :param x: input vector Nx1
    :return: vector of sigmoid function values calculated for elements x, Nx1
    '''
    return 1 / (1 + np.exp(-x))

def logistic_cost_function(w, x_train, y_train):
    '''
    :param w: model parameters Mx1
    :param x_train: training set features NxM
    :param y_train: training set labels Nx1
    :return: function returns tuple (val, grad), where val is a velue of logistic function and grad its gradient over w
    '''
    sigma = sigmoid(x_train @ w)
    N = y_train.shape[0]
    p_D_w = np.prod((sigma ** y_train) * ((1 - sigma) ** (1 - y_train)))
    L_w = -np.log(p_D_w) / N
    grad = - (x_train.transpose() @ (y_train - sigma)) / N
    return L_w, grad


def gradient_descent(obj_fun, w0, epochs, eta):
    '''
    :param obj_fun: objective function that is going to be minimized (call val,grad = obj_fun(w)).
    :param w0: starting point Mx1
    :param epochs: number of epochs / iterations of an algorithm
    :param eta: learning rate
    :return: function optimizes obj_fun using gradient descent. It returns (w,func_values),
    where w is optimal value of w and func_valus is vector of values of objective function [epochs x 1] calculated for each epoch
    '''
    w = w0
    wA = []
    _, grad = obj_fun(w0)
    for i in range(epochs):
        w = w - eta * grad
        val, grad = obj_fun(w)
        wA.append(val)
    return w, np.array(wA).reshape(epochs, 1)


def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    '''
    :param obj_fun: objective function that undergoes optimization. Call val,grad = obj_fun(w,x,y), where x,y indicates mini-batches.
    :param x_train: training data (feature vectors)NxM
    :param y_train: training data (labels) Nx1
    :param w0: starting point Mx1
    :param epochs: number of epochs
    :param eta: learning rate
    :param mini_batch: size of mini-batches
    :return: function optimizes obj_fun using gradient descent. It returns (w,func_values),
    where w is optimal value of w and func_valus is vector of values of objective function [epochs x 1] calculated for each epoch. V
    Values of func_values are calculated for entire training set!
    '''
    N = y_train.shape[0]
    M = N // mini_batch
    # splitting array for small pieces to use stochastic gradient_descent alghorithm
    x_mini_batch = np.vsplit(x_train, M)
    y_mini_batch = np.vsplit(y_train, M)

    w = w0
    wA = []
    for i in range(epochs):
        for x, y in zip(x_mini_batch, y_mini_batch):
            _, grad = obj_fun(w, x, y)
            w = w - eta * grad
        wA.append(obj_fun(w, x_train, y_train)[0])
    return w, np.array(wA).reshape(epochs, 1)


def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    '''
    :param w: model parameters Mx1
    :param x_train: training set (features) NxM
    :param y_train: training set (labels) Nx1
    :param regularization_lambda: regularization parameters
    :return: function returns tuple(val, grad), where val is a velue of logistic function with regularization l2,
    and grad its gradient over w
    '''
    w_0 = np.copy(w)
    w_0[0] = 0
    N = y_train.shape[0]
    sigma = sigmoid(x_train @ w)
    p_D_w = np.prod((sigma ** y_train) * ((1 - sigma) ** (1 - y_train)))
    L_w_l = -np.log(p_D_w) / N + regularization_lambda / 2 * np.linalg.norm(w_0) ** 2
    grad = - (x_train.transpose() @ (y_train - sigma)) / N + regularization_lambda * w_0
    return L_w_l, grad


def prediction(x, w, theta):
    '''
    :param x: observation matrix NxM
    :param w: parameter vector Mx1
    :param theta: classification threshold [0,1]
    :return: function calculates vector y Nx1. Vector is composed of labels {0,1} for observations x
     calculated using model (parameters w) and classification threshold theta
    '''
    return np.apply_along_axis(arr=sigmoid(x @ w), func1d=lambda arg: arg >= theta, axis=1)


def f_measure(y_true, y_pred):
    '''
    :param y_true: vector of ground truth labels Nx1
    :param y_pred: vector of predicted labels Nx1
    :return: value of F-measure
    '''
    N = y_true.shape[0]
    TP = np.count_nonzero(y_true & y_pred)
    F = np.count_nonzero(np.bitwise_xor(y_true, y_pred))
    return 2 * TP / (2 * TP + F)


def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    '''
    :param x_train: trainig set (features) NxM
    :param y_train: training set (labels) Nx1
    :param x_val: validation set (features) Nval x M
    :param y_val: validation set (labels) Nval x 1
    :param w0: vector of initial values of w
    :param epochs: number of iterations of SGD
    :param eta: learning rate
    :param mini_batch: mini-batch size
    :param lambdas: list of lambda values that have to be considered in model selection procedure
    :param thetas: list of theta values that have to be considered in model selection procedure
    :return: Functions makes a model selection. It returs tuple (regularization_lambda, theta, w, F), where regularization_lambda
    is the best velue of regularization parameter, theta is the best classification threshold, and w is the best model parameter vector.
    Additionally function returns matrix F, which stores F-measures calculated for each pair (lambda, theta).
    Use SGD and training criterium with l2 regularization for training.
    '''
    F = np.zeros(shape=(len(lambdas), len(thetas)))
    max_f = [0, 0, 0, -1]
    for i, λ in enumerate(lambdas):
        obj_fun = lambda w, x, y: regularized_logistic_cost_function(w, x, y, λ)
        w, func_values = stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch)
        for j, θ in enumerate(thetas):
            f = f_measure(y_val, prediction(x_val, w, θ))
            F[i, j] = f
            if f > max_f[3]: max_f = [λ, θ, w, f]
    max_f[3] = F
    return tuple(max_f)
