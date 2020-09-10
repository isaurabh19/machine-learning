from source.problem4 import *
from sklearn.metrics import roc_auc_score
import statistics


def run(dataset_tuple):
    train_data, test_data = dataset_tuple
    weights = log_reg(train_data[:, :-1], train_data[:, -1:])
    predictions = get_predictions(weights, test_data[:, :-1])
    auroc = roc_auc_score(test_data[:, -1:], predictions)
    return auroc


for dataset in get_datasets():
    ten_pairs = get_10_pairs(dataset)
    auroc_scores = list(map(run, ten_pairs))
    mean_auroc = statistics.mean(auroc_scores)

    print("Mean AUROC curve: {}".format(mean_auroc))
