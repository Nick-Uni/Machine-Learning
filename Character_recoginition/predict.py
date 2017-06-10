import pickle as pkl
import numpy as np
import time


def train():
    x_train, y_train = pkl.load(open('train.pkl', mode='rb'))
    x_train = x_train[0:17500]
    y_train = y_train[0:17500]
    return x_train, y_train

def validate():
    x_validate, y_validate = pkl.load(open('train.pkl', mode='rb'))
    x_validate = x_validate[27500:30000]
    y_validate = y_validate[27500:30000]
    return x_validate, y_validate


def hamming_distance(X, X_train):
    """
        :param X: set of objects that are going to be compared N1xD
        :param X_train: set of objects compared against param X N2xD
        Functions calculates Hamming distances between all objects from set X  and all object from set X_train.
        Resulting distances are returned as matrices.
        :return: Distance matrix between objects X and X_train N1xN2
    """
    X_train = np.transpose(X_train)
    NOT_X = np.subtract(np.ones(shape=(X.shape[0], X.shape[1])), X)
    NOT_X_train = np.subtract(np.ones(shape=(X_train.shape[0], X_train.shape[1])), X_train)
    hamming = X @ NOT_X_train + NOT_X @ X_train
    return hamming


def sort_train_labels_knn(Dist, y):
    """
        Function sorts labels of training data y accordingly to probabilities stored in matrix Dist.
        Function returns matrix N1xN2. In each row there are sorted data labels y accordingly to corresponding row of matrix Dist.
        :param Dist: Distance matrix between objects X and X_train N1xN2
        :param y: N2-element vector of labels
        :return: Matrix of sorted class labels ( use metgesort algorithm)
    """
    sorted = y[Dist.argsort(kind='mergesort', axis=1)]
    # sort = time.time()
    # print('{}: {}'.format('sort', sort - start))
    return sorted


def p_y_x_knn(y, k):
    """
        Function calculates conditional probability p(y|x) for
        all classes and all objects from test set using KNN classifier
        :param y: matrix of sorted labels for training set N1xN2 <-------Distance Matrix
        :param k: number of nearest neighbours
        :return: matrix of probabilities for objects X
    """

    resizedArray = np.delete(y, range(k, y.shape[1]), axis=1)
    probabilities = np.apply_along_axis(np.bincount, axis=1, arr=resizedArray, minlength=36)
    probabilities = np.delete(probabilities, 0, axis=1)
    # knn = time.time()
    # print('{}: {}'.format('knn', knn - start))
    return probabilities

def predict(x):
    """
    	Function takes a matrix of images as an argument (NxD). Each row of the matrix represents a single image.
    	Function returns vector y (Nx1). Each element is a number in range {0, ..., 35}, which determines symbol recognized in the image.
        :param x: matrix NxD
        :return: vector Nx1
    """
    x_train, y_train = train()[0], train()[1]
    distance = hamming_distance(x, x_train)
    # hamm = time.time()
    # print('{}: {}'.format('hamm', hamm - start))
    prediction = np.argmax(p_y_x_knn(sort_train_labels_knn(distance, y_train), 1), axis=1) + 1
    return prediction




if __name__ == "__main__":
    start = time.time()
    x_val, y_val = validate()[0], validate()[1]
    a = predict(x_val)
    print(a)
    print(np.count_nonzero(a==y_val)/2500)
    finish = time.time()
    print('{}: {}'.format('FINISH!', finish - start))
