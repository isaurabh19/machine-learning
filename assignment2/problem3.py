import numpy as np
from sklearn.datasets import make_regression
from matplotlib import pyplot


def sse(X, Y, initial_w, X_transpose):
    max_iterations = 10000
    t = 0
    print("Training start for SSE")
    while t < max_iterations:
        error = list(map(lambda x: np.subtract(np.matmul(np.transpose(initial_w), x[0]), x[1]), zip(X, Y)))
        w_updates = list(map(lambda x: np.matmul(x, np.transpose(error)), X_transpose))
        w_updates = np.multiply(w_updates, 2 * learning_rate)
        new_w = np.subtract(initial_w, w_updates)

        delta = np.linalg.norm(np.subtract(new_w, initial_w))
        if delta <= min_delta:
            break

        # if t % 10 == 0:
        #     print("Old Ws = {} and new Ws = {} and delta = {}".format(initial_w, new_w, delta))

        initial_w = new_w
        t += 1

    print("Training done: SSE")

    return new_w


def r_squared(predictions, Y):
    mean_y = np.mean(Y)
    denom = np.sum(np.subtract(mean_y, Y) ** 2)
    num = np.sum(np.subtract(predictions, Y) ** 2)
    return 1 - (num / denom)


def euclidean(X, Y, X_transpose):
    max_iterations = 100000
    t = 0
    initial_w = np.divide(W, 1.09)
    # learning_rate = 0.0002
    print("Training start for Orthogonal Dist with initial weights {}".format(initial_w))
    # initial_w = initial_w[1:]
    while t < max_iterations:
        initial_w0 = initial_w[0]
        w_squared = np.sum(initial_w[1:] ** 2)
        error = list(map(lambda x: np.subtract(np.matmul(np.transpose(initial_w), x[0]), x[1]), zip(X, Y)))
        error_square = list(map(lambda x: x ** 2, error))
        sum_error_square = sum(error_square)
        sum_err_sq_w = list(map(lambda w: sum_error_square * w, initial_w))
        error_Xi = list(map(lambda x: np.matmul(x, np.transpose(error)), X_transpose))
        w_updates = np.multiply(2 / w_squared,
                                np.subtract(error_Xi, np.divide(sum_err_sq_w, w_squared)))
        new_w = np.subtract(initial_w, np.multiply(w_updates, learning_rate))
        new_w[0] = initial_w0 - ((2 * learning_rate / w_squared) * sum(error))

        delta = np.linalg.norm(np.subtract(new_w, initial_w))
        if delta <= min_delta:
            break

        if t % 1000 == 0:
            print("Old Ws = {} and new Ws = {} and delta = {}".format(initial_w, new_w, delta))

        initial_w = new_w
        t += 1

    print("Training done for Orthogonal Dist")
    return new_w


n_samples = 500
features = 5
learning_rate = 0.0001
min_delta = 0.0001
bias = 1.5
for i in range(0, 5):
    X, Y, W = make_regression(n_samples, features - 1, noise=3, bias=bias, coef=True, random_state=i)
    X = np.insert(X, 0, 1, axis=1)
    W = np.insert(W, 0, bias)
    print("W used for generation {}".format(W))
    initial_W = np.ones(features)
    X_T = np.transpose(X)

    sse_w = sse(X, Y, initial_W, X_T)
    sse_predictions = np.matmul(X, np.transpose(sse_w))
    r_2_sse = r_squared(sse_predictions, Y)

    orthogonal_w = euclidean(X, Y, X_T)
    orthogonal_predictions = np.matmul(X, np.transpose(orthogonal_w))
    r_2_orthogonal = r_squared(orthogonal_predictions, Y)

    X_T_X = np.matmul(X_T, X)
    inv_X_T_X = np.linalg.inv(X_T_X)
    X_inv_X_T_X = np.matmul(X, inv_X_T_X)
    X_inv_X_T_X_XT = np.matmul(X_inv_X_T_X, X_T)
    ML_predictions = np.matmul(X_inv_X_T_X_XT, Y)
    r_2_ml = r_squared(ML_predictions, Y)

    print("R squared for SSE {}".format(r_2_sse))
    print("R squared for Orthogonal {}".format(r_2_orthogonal))
    print("R squared for ML {}".format(r_2_ml))

    print("Actual W ={}".format(W))
    print("SSE W ={}".format(sse_w))
    print("Ortho W={}".format(orthogonal_w))
