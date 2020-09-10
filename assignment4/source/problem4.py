from sklearn.datasets import load_breast_cancer
import numpy as np
import math


def get_datasets():
    data1, target1 = load_breast_cancer()
    dataset1 = np.concatenate((data1, np.array([target1].T)), axis=1)
    np.random.shuffle(dataset1)

    dataset2 = np.loadtxt("../data/transfusion.csv", delimiter=",")
    np.random.shuffle(dataset2)

    dataset3 = np.loadtxt("../data/data_banknote_authentication.csv", delimiter=",")
    np.random.shuffle(dataset3)

    return dataset1, dataset2, dataset3


def get_10_pairs(dataset):
    n_folds = split_n_folds(dataset)
    ten_pairs = []
    for i in range(10):
        test_data = n_folds[i]
        train_data = []
        for j in range(10):
            if j != i:
                train_data.extend(n_folds[j])
        ten_pairs.append((train_data, test_data))
    return ten_pairs


# data2 = dataset2[:, :-1]
# target2 = dataset2[:, -1:]
#
#
#
# data3 = dataset3[:, :-1]
# target3 = dataset3[:, -1:]

delta = 0.0001


def split_n_folds(dataset):
    block_size = int(dataset / 10)
    folds = []
    for i in range(9):
        folds.append(dataset[i * block_size: (i + 1) * block_size, :])
    folds.append(dataset[9 * block_size:, :])
    return folds


def log_reg(data, target):
    data_t = np.transpose(data)
    initial_weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(data_t, data)), data_t), target)
    t = 0

    while t < 10000:
        new_weights = initial_weights + np.matmul(np.linalg.inv(hessian(initial_weights, data)),
                                                  gradient(data, target, initial_weights))
        norm_initial_weights = np.divide(initial_weights, np.sum(np.abs(initial_weights)))
        norm_new_weights = np.divide(new_weights, np.sum(np.abs(new_weights)))
        eps = np.sum(np.abs(np.subtract(norm_new_weights, norm_initial_weights)))

        if eps <= delta:
            break

        if t % 1000 == 0:
            print("Iteration : {} Old weights : {}, and new weights : {} , delta : {}".format(t, initial_weights,
                                                                                              new_weights, eps))

        initial_weights = new_weights
        t += 1

    return initial_weights


def get_predictions(weights, test_data):
    prediction_scores = sigmoid(weights, test_data)
    prediction_label = list(map(lambda x: 1 if (x >= 0.5) else 0, prediction_scores))

    return prediction_label


def pca(data):
    """

    :param data: nxd
    :return: nxl
    """
    # cov matrix should be a dxd matrix thus first transpose nxd matrix to dxn
    data = data.T
    cov_matrix = np.cov(data)
    eig_val, eig_vec = np.linalg.eig(cov_matrix)
    sorted_eigen_values = -np.sort(-1 * eig_val)
    sorted_eigen_vectors = eig_vec[:, np.argsort(-1 * eig_val)]

    retained_eigenvectors = retained_features(sorted_eigen_values, sorted_eigen_vectors)

    transformed_data = retained_eigenvectors.T.dot(data)
    return transformed_data.T


def retained_features(eigenvalues, eigenvectors):
    """

    :param eigenvalues:
    :param eigenvectors:
    :return: dxl matrix
    """
    current_variance = 0
    total_variance = np.sum(eigenvalues)
    for i, value in enumerate(eigenvalues):
        current_variance += value
        if current_variance / total_variance >= 0.99:
            return eigenvectors[:, :i + 1]

    return eigenvectors


def sigmoid(weights, data):
    """

    :param weights: 1xd
    :param data: nxd
    :return: nx1
    """
    prediction_scores = list(map(lambda x: 1 / (1 + math.exp(-np.matmul(np.transpose(weights), x))), data))
    return prediction_scores


def gradient(data, target, weights):
    error = np.subtract(target, sigmoid(data, weights))
    gradients = np.matmul(np.transpose(data), error)
    return gradients


def hessian(weights, data):
    prediction_scores = sigmoid(weights, data)
    P = np.diag(prediction_scores)
    identity = np.identity(data.shape[0])
    I_P = np.subtract(identity, P)
    P_IP = np.matmul(P, I_P)
    temp = np.matmul(-np.transpose(data), P_IP)
    hessian_matrix = np.matmul(temp, data)
    return hessian_matrix


def z_score(column):
    mean = np.mean(column)
    sd = np.std(column)
    return list(map(lambda x: (x - mean) / sd, column))


def zero_mean(column):
    mean = np.mean(column)
    return list(map(lambda x: x - mean, column))
