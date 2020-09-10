from source.problem4 import *
from sklearn.metrics import roc_auc_score
import statistics
import numpy as np


def run(dataset_tuple):
    train_data, test_data = dataset_tuple
    normalized_train_data = get_normalized(train_data[:, :-1])
    normalized_test_data = get_normalized(test_data[:, :-1])
    weights = log_reg(normalized_train_data[:, :-1], normalized_train_data[:, -1:])
    predictions = get_predictions(weights, normalized_test_data[:, :-1])
    auroc = roc_auc_score(normalized_test_data[:, -1:], predictions)
    return auroc


def get_normalized(train_data):
    normalized_data = np.apply_along_axis(z_score, 0, train_data)
    return normalized_data


for dataset in get_datasets():
    ten_pairs = get_10_pairs(dataset)
    auroc_scores = list(map(run, ten_pairs))
    mean_auroc = statistics.mean(auroc_scores)

    print("Mean AUROC curve: {}".format(mean_auroc))
