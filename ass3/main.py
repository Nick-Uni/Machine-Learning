# --------------------------------------------------------------------------
# ----------------  System Analysis and Decision Making --------------------
# --------------------------------------------------------------------------
#  Assignment 1: Logistic regression
#  Authors: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ----------------------- DO NOT MODIFY THIS FILE --------------------------
# -------------------------------------------------------------------------

import functools
import warnings
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib import animation
import pickle
import numpy as np
from utils import hog
from time import sleep
from test import TestRunner
from content import (sigmoid, logistic_cost_function, gradient_descent,
                              stochastic_gradient_descent, regularized_logistic_cost_function,
                              prediction, f_measure, model_selection)

PICKLE_FILE_PATH = 'data.pkl'
TEST_FILE_PATH = 'test_data.pkl'
EPOCHS = 100
MINIBATCH_SIZE = 50
PATCH_WIDTH = 92
PATCH_HEIGHT = 112
STEP = 20
marker_positions = []

def load_test():
    with open(TEST_FILE_PATH, 'rb') as f:
        return pickle.load(f)


def load_data():
    with open(PICKLE_FILE_PATH, 'rb') as f:
        return pickle.load(f)


def plot_f_values(f1, f2):
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'none'
    plt.style.use(['dark_background'])
    xs = range(len(f1))

    plt.title("Comparison of GD and SGD algorithms")
    plt.ylabel('Values of optimization objective')
    plt.xlabel('Iteration ID')

    gd_line, = plt.plot(xs, f1, 'r-', color='#FFCC55', label='GD')
    sgd_line, = plt.plot(xs, f2, 'r-', color='#FF5533', label='SGD')
    plt.legend(handles=[gd_line, sgd_line])
    fig = plt.gcf()
    fig.canvas.set_window_title('Values of objective function for GD and SGD')
    plt.draw()
    plt.waitforbuttonpress(0)


def plot_theta_lambda(F_vals, theta_vals, lambda_vals):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(F_vals)
    fig.colorbar(cax)

    ax.set_xticklabels([''] + theta_vals)
    ax.set_yticklabels([''] + lambda_vals)
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.set_tick_params(labelbottom=True, labeltop=False, top=False)
    ax.set_xlabel(r'Classification threshold $\theta$')
    ax.set_ylabel('Regularization coefficient $\lambda$')
    plt.title("Model selection for logistic regression")
    plt.axis('tight')
    fig = plt.gcf()
    fig.canvas.set_window_title('Values of F-measure on validation set')
    plt.draw()
    plt.waitforbuttonpress(0)

def face_detect_patch(patch, w, theta):
    patch = patch / 255.0
    hog_patch = hog(patch)
    return prediction(np.transpose(np.concatenate(([[1]], hog_patch))), w, theta)


def get_patch(img, x, y):
    x1, x2 = x, x + PATCH_WIDTH
    y1, y2 = y, y + PATCH_HEIGHT
    return img[y1: y2, x1: x2]


def animate(i, ax, patch, patch_positions, img, w, theta):
    markers = []
    for position in marker_positions:
        markers.append(patches.Rectangle(position, PATCH_WIDTH, PATCH_HEIGHT, fill=False, color='g',linewidth=3))

    for marker in markers:
        ax.add_patch(marker)
    if i > len(patch_positions) - 1:
        return [patch] + markers

    img_size = img.shape
    x, y = patch_positions[i]
    patch.set_xy([x, img_size[0] - PATCH_HEIGHT - y])

    cut_out = get_patch(img, x, y)

    if face_detect_patch(cut_out, w, theta):
        marker_position = [x, img_size[0] - PATCH_HEIGHT - y]
        marker_positions.append(marker_position)
        marker = patches.Rectangle(marker_position, PATCH_WIDTH, PATCH_HEIGHT, fill=False, color='g',linewidth=3)
        ax.add_patch(marker)
        markers.append(marker)

    return [patch] + markers


def animate_face_detect(w, theta):
    marker_positions = []
    img = mpimg.imread('image2017.jpg')
    fig = plt.figure(figsize=(7.5, 4.23))
    plt.axis('equal')
    plt.tight_layout()
    ax = fig.add_subplot(111)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(0, img.shape[0])
    ax.xaxis.set_tick_params(labelbottom=False, labeltop=False, top=False, bottom=False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)

    patch = patches.Rectangle((PATCH_WIDTH, PATCH_HEIGHT), 100, 120, fill=False, color='r',linewidth=3)

    ax.imshow(img, extent=[0, img.shape[1], 0, img.shape[0]], aspect='auto', interpolation="bicubic")
    ax.add_patch(patch)
    patch_positions = []

    for y in range(0, img.shape[0] - PATCH_HEIGHT, STEP):
        for x in range(0, img.shape[1] - PATCH_WIDTH, STEP):
            patch_positions.append([x, y])

    fargs = [ax, patch, patch_positions, img, w, theta]
    anim = animation.FuncAnimation(fig, animate,
                                   fargs=fargs,
                                   #init_func=self.initialize_animation,
                                   frames=200,
                                   interval=2,
                                   blit=True, repeat=False)
    plt.draw()
    plt.waitforbuttonpress(0)


def run_unittests():
    test_runner = TestRunner()
    results = test_runner.run()
    if results.failures or results.errors:
        exit()
    sleep(0.1)


def run_training():

    data = load_data()

    print('---------- Training logistic regression with the use of gradient descent --------')
    print('------------------ May take up to 1 min. -----------------------------')

    eta = 0.1
    theta = 0.65
    lambdas = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    thetas = list(np.arange(0.1, 0.9, 0.05))

    log_cost_for_data = functools.partial(logistic_cost_function, x_train=data['x_train'], y_train=data['y_train'])
    w_0 = np.zeros([data['x_train'].shape[1], 1])
    w_computed1, f_values1 = gradient_descent(log_cost_for_data, w_0, EPOCHS, eta)

    print('Final value of optimization objective: {:.4f}'.format(f_values1[-1][0]))

    print('\n-------  Training logistic regression with the use of stochastic gradient descent  -----')
    print('------------------To może potrwać ok. 1 min -----------------------------')

    w_0 = np.zeros([data['x_train'].shape[1], 1])
    w_computed2, f_values2 = stochastic_gradient_descent(logistic_cost_function, data['x_train'],
                                                       data['y_train'], w_0, EPOCHS, eta, MINIBATCH_SIZE)

    print('Final value of optimization objective: {:.4f}'.format(f_values2[-1][0]))
    print('\n--- Press any key to continue ---')
    plot_f_values(f_values1, f_values2)

    print('\n----------------------- Model selection -------------------------------')
    print('--Optimization method: SGD--')
    print('--Training criterium: regularized_logistic_cost_function--')
    print('--Step: {}--'.format(eta))
    print('--Number of epochs: {}--'.format(EPOCHS))
    print('--Mini-batch size: {}--'.format(MINIBATCH_SIZE))

    w_0 = np.zeros([data['x_train'].shape[1], 1])
    l, t, w_computed, F = model_selection(data['x_train'], data['y_train'], data['x_val'],
                                          data['y_val'], w_0, EPOCHS, eta, MINIBATCH_SIZE, lambdas, thetas)

    print('The best regulatization coefficient: {}'.format(l))
    print('The best threshold value (theta): {:.4f}'.format(t))
    print('The best value of F-measure: {:.4f}'.format(np.max(F)))
    print('\n--- Press any key to continue ---')
    plot_theta_lambda(F, thetas, lambdas)

    print('\n------------------------ FACE DETECTION -------------------------------\n')
    animate_face_detect(w_computed, t)



if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    run_unittests()

    run_training()