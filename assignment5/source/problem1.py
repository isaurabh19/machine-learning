from keras.callbacks import EarlyStopping
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import source.utils as ut
import numpy as np
import statistics
import math
import tensorflow as tf
import time

tf.logging.set_verbosity(tf.logging.ERROR)


def implement_bootstrap(data, train_index, test_index, h1, h2):
    model = utils.build_nn(h1, h2)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=3)
    aurocs = []
    accuracies = []
    X = data[:, :-1]
    y = data[:, -1:]
    model.fit(X[train_index], y[train_index], epochs=1000, callbacks=[early_stopping], verbose=0)
    for i in range(100):
        boot_test_data = resample(data[test_index], replace=True,
                                  n_samples=500)  # TODO: remove stratify and instead resample 500
        X_boot = boot_test_data[:, :-1]
        y_boot = boot_test_data[:, -1:]
        accuracy = model.evaluate(X_boot, y_boot)[1]
        y_pred_score = model.predict(X_boot)
        auroc = roc_auc_score(y_boot, y_pred_score)

        accuracies.append(accuracy)
        aurocs.append(auroc)

    # accuracies = np.multiply(accuracies, 100)
    return accuracies, aurocs

    # boot_mean_acc = np.mean(accuracies)
    # standard_error_acc = math.sqrt(sum(list(map(lambda x: (x - boot_mean_acc) ** 2, accuracies))) / 99)
    #
    # boot_mean_auroc = statistics.mean(aurocs)
    # standard_error_auroc = math.sqrt(sum(list(map(lambda x: (x - boot_mean_auroc) ** 2, aurocs))) / 99)
    #
    # return boot_mean_acc, standard_error_acc, boot_mean_auroc, standard_error_auroc


def run_parallely(h1, h2, data):
    X = data[:, :-1]
    y = data[:, -1:]
    skf = StratifiedKFold(n_splits=10)
    indices = [(train_index, test_index) for train_index, test_index in skf.split(X, y)]
    results = Parallel(n_jobs=8, verbose=10)(
        delayed(implement_bootstrap)(data, index[0], index[1], h1, h2) for index in indices)

    all_accuracies = []
    all_aurocs = []
    for result in results:
        all_accuracies.append(result[0])
        all_aurocs.append(result[1])

    all_accuracies_matrix = np.array(all_accuracies)
    all_aurocs_matrix = np.array(all_aurocs)

    all_accuracies = np.mean(all_accuracies_matrix, axis=0)
    all_aurocs = np.mean(all_aurocs_matrix, axis=0)

    print("Length {} {}".format(len(all_accuracies), len(all_aurocs)))
    boot_mean_acc = np.mean(all_accuracies)
    standard_error_acc = math.sqrt(sum(list(map(lambda x: (x - boot_mean_acc) ** 2, all_accuracies))) / 99)

    boot_mean_auroc = statistics.mean(all_aurocs)
    standard_error_auroc = math.sqrt(sum(list(map(lambda x: (x - boot_mean_auroc) ** 2, all_aurocs))) / 99)

    return boot_mean_acc, standard_error_acc, boot_mean_auroc, standard_error_auroc
    # results = []
    # for n, index in enumerate(indices):
    #     results.append(implement_bootstrap(data, index[0], index[1], h1, h2))
    #     print("Done fold {}/10".format(n + 1))

    # return results


datasets = ut.Utils().get_dataset(1000)
i = 1
for data in datasets:
    for combinations in [(1, 0), (4, 0), (8, 0), (1, 3), (4, 3), (8, 3)]:
        utils = ut.Utils()
        start = time.time()
        results = run_parallely(combinations[0], combinations[1], data)
        boot_accuracies, boot_acc_se, boot_aurocs, boot_auroc_se = results
        print("Concept {} Combination {}x{} Sample mean accuracy {} and Standard error accuracy {}".format(i,
                                                                                                           combinations[
                                                                                                               0],
                                                                                                           combinations[
                                                                                                               1],
                                                                                                           boot_accuracies,
                                                                                                           boot_acc_se))

        print("Concept {} Combination {}x{} Sample mean auroc {} and standard error auroc {}".format(i, combinations[0],
                                                                                                     combinations[1],
                                                                                                     boot_aurocs,
                                                                                                     boot_auroc_se))

        print("-----------Total runtime {}------------".format((time.time() - start) / 60))
    i += 1
