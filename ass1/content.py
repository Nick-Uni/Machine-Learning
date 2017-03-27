# --------------------------------------------------------------------------
# ----------------  System Analysis and Decision Making --------------------
# --------------------------------------------------------------------------
#  Assignment 1: Linear regression
#  Authors: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np
from numpy.linalg import inv
from utils import polynomial

def mean_squared_error(x, y, w):
    mse = np.mean((y - polynomial(x, w)) ** 2)
    return mse


def design_matrix(x_train, M):
    matrix = np.zeros((len(x_train), M + 1))
    row = 0
    for x in x_train:
        for i in range(M + 1):
            matrix[row, i] = x ** i
        row += 1
    return matrix



def least_squares(x_train, y_train, M):
    '''
    :param x_train: training input vector  Nx1
    :param y_train: training output vector Nx1
    :param M: polynomial degree
    :return: tuple (w,err), where w are model parameters and err mean squared error of fitted polynomial
    '''
    m = design_matrix(x_train, M)
    transp = m.transpose()
    mul = np.matmul(transp,m)
    w = np.matmul((np.matmul(inv(mul), m.transpose())), y_train)
    return (w, mean_squared_error(x_train, y_train, w))


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param M: polynomial degree
    :param regularization_lambda: regularization parameter
    :return: tuple (w,err), where w are model parameters and err mean squared error of fitted polynomial with l2 regularization
    '''
    m = design_matrix(x_train, M)
    transp = m.transpose()
    squared = np.matmul(transp, m)
    size = squared.shape[0]
    mul = np.matmul(transp, m)
    lam = regularization_lambda * np.identity(size)
    w = np.matmul((np.matmul(inv(mul + lam), transp)), y_train)
    return (w, mean_squared_error(x_train, y_train, w))



def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param x_val: validation input vector Nx1
    :param y_val: validation output vector Nx1
    :param M_values: array of polynomial degrees that are going to be tested in model selection procedure
    :return: tuple (w,train_err, val_err) representing model with the lowest validation error
    w: model parameters, train_err, val_err: training and validation mean squared error
    '''
    w_list = []
    train = []
    val = []
    for i in M_values:
        (w, err) = least_squares(x_train, y_train, i)
        w_list.append(w)
        train.append(err)
        val.append(mean_squared_error(x_val, y_val, w))
    index = val.index(min(val))

    return (w_list[index], train[index], val[index])


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param x_val: validation input vector Nx1
    :param y_val: validation output vector Nx1
    :param M: polynomial degree
    :param lambda_values: array of regularization coefficients are going to be tested in model selection procedurei
    :return:  tuple (w,train_err, val_err, regularization_lambda) representing model with the lowest validation error
    (w: model parameters, train_err, val_err: training and validation mean squared error, regularization_lambda: the best value of regularization coefficient)
    '''

    w_list = []
    train = []
    val = []
    for lambd in lambda_values:
        (w, err) = regularized_least_squares(x_train, y_train, M, lambd)
        w_list.append(w)
        train.append(err)
        val.append(mean_squared_error(x_val, y_val, w))
    index = val.index(min(val))

    return (w_list[index], train[index], val[index], lambda_values[index])