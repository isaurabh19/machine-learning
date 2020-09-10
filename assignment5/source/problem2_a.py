from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils import resample
from keras.models import load_model
from joblib import Parallel, delayed
from scipy import stats
import source.utils as ut
import numpy as np
import time
import statistics
import math
import copy


# trained_models = []


def train_nn_parallely(config, data_fold, train_index, h1, h2, problem, utils):
    data_fold = data_fold[train_index]
    if problem == 2:
        data_fold = resample(data_fold, replace=True, n_samples=1000)
    np.random.shuffle(data_fold)
    X_train = data_fold[:, :-1]
    y_train = data_fold[:, -1:]
    model = utils.build_nn(h1, h2, loss=config[1], kernel_initiliazer=config[0])
    model.fit(X_train, y_train, epochs=500, verbose=0)
    # trained_models.append(model)
    # model.save("{}_{}_{}_{}model_{}_{}_{}.h5".format(config[1][0], config[1][1], config[2], config[0], h1, h2, i))
    # return "{}_{}_{}_{}model_{}_{}_{}.h5".format(config[1][0], config[1][1], config[2], config[0], h1, h2, i)
    return model


def single_bootstrap(boot_test_data, trained_models, problem):  # TODO add the trained_models param back
    # start_boot = time.time()
    X_boot = boot_test_data[:, :-1]
    y_boot = boot_test_data[:, -1:]
    scores = np.empty((len(X_boot), 1), float)

    for model in trained_models:
        y_pred_score = model.predict(X_boot)
        scores = np.column_stack((scores, y_pred_score))

    scores = scores[:, 1:]
    # For each datapoint in scores, average the score i.e averaging output of each network
    y_prob_average_score = np.mean(scores, axis=1)  # average_score.shape = len(y_boot),1
    pred_labels = list(
        map(lambda x: 1 if x >= 0.5 else 0, y_prob_average_score))  # TODO use median of scores instead of 0.5

    # single values
    accuracy = accuracy_score(y_boot, pred_labels)
    auroc = roc_auc_score(y_boot, y_prob_average_score)

    # print("Time for single bootstrap run {}".format((time.time() - start_boot) / 60))
    return accuracy, auroc


def run_kfold_parallely(h1, h2, data_og, train_index, test_index, configs, problem):
    # X = data_og[:, :-1]
    # y = data_og[:, -1:]
    utils = ut.Utils()
    start = time.time()
    models = [train_nn_parallely(config, data_og, train_index, h1, h2, problem, utils) for config in configs]
    print("training time {} for {}x{}".format((time.time() - start) / 60, h1, h2))
    # models = Parallel(n_jobs=8, verbose=2)(
    #     delayed(train_nn_parallely)(config, data_og, train_index, h1, h2) for config in configs)
    # resampled_test_data = [resample(data_og[test_index], replace=True, stratify=y[test_index]) for _ in range(10)]
    bootstrap_results = []
    for b in range(100):
        print("Bootstrap : {}/100".format(b + 1))
        boot_test_data = resample(data_og[test_index], replace=True, n_samples=500)
        bootstrap_results.append(single_bootstrap(boot_test_data, models, problem))
    # bootstrap_results = Parallel(n_jobs=8, verbose=1)(
    #     delayed(single_bootstrap)(boot_test_data) for boot_test_data in resampled_test_data)
    accuracies, aurocs = list(zip(*bootstrap_results))

    return accuracies, aurocs

    # sample_mean_acc = statistics.mean(accuracies)
    # standard_error_acc = math.sqrt(sum([(x - sample_mean_acc) ** 2 for x in accuracies]) / 99)
    #
    # sample_mean_auroc = statistics.mean(aurocs)
    # standard_error_auroc = math.sqrt(sum([(x - sample_mean_auroc) ** 2 for x in aurocs]) / 99)

    # return sample_mean_acc, standard_error_acc, sample_mean_auroc, standard_error_auroc


datasets = ut.Utils().get_dataset(1000)
i = 1
configs = ut.Utils().get_nn_configs(10)
problem = 2
for data in datasets:
    # if i == 1:
    #     i += 1
    #     continue
    start = time.time()
    for combinations in [(1, 0), (4, 0), (8, 0), (1, 3), (4, 3), (8, 3)]:
        # run_kfolds_serially(combinations[0], combinations[1], data)
        X = data[:, :-1]
        y = data[:, -1:]
        skf = StratifiedKFold(n_splits=10)
        indices = [(train_index, test_index) for train_index, test_index in skf.split(X, y)]
        results = Parallel(n_jobs=8, verbose=1)(
            delayed(run_kfold_parallely)(combinations[0], combinations[1], data, index[0], index[1], configs, problem)
            for index in indices)
        all_accuracies = []
        all_aurocs = []
        for result in results:
            all_accuracies.append(result[0])
            all_aurocs.append(result[1])

        all_accuracies_matrix = np.array(all_accuracies)
        all_aurocs_matrix = np.array(all_aurocs)

        all_accuracies = np.mean(all_accuracies_matrix, axis=0)
        all_aurocs = np.mean(all_aurocs_matrix, axis=0)

        boot_mean_acc = np.mean(all_accuracies)
        standard_error_acc = math.sqrt(sum(list(map(lambda x: (x - boot_mean_acc) ** 2, all_accuracies))) / 99)

        boot_mean_auroc = statistics.mean(all_aurocs)
        standard_error_auroc = math.sqrt(sum(list(map(lambda x: (x - boot_mean_auroc) ** 2, all_aurocs))) / 99)

        # boot_accuracies, boot_acc_se, boot_aurocs, boot_auroc_se = list(zip(*results))
        print("Concept {} Combination {}x{} Sample mean accuracy {} and Standard error accuracy {}".format(i,
                                                                                                           combinations[
                                                                                                               0],
                                                                                                           combinations[
                                                                                                               1],
                                                                                                           boot_mean_acc,
                                                                                                           standard_error_acc))

        print("Concept {} Combination {}x{} Sample mean auroc {} and standard error auroc {}".format(i, combinations[0],
                                                                                                     combinations[1],
                                                                                                     boot_mean_auroc,
                                                                                                     standard_error_auroc))

    print("-----------Total runtime in minutes {}------------".format((time.time() - start) / 60))
    i += 1

# def run_kfolds_serially(h1, h2, data_og):
#     X = data_og[:, :-1]
#     y = data_og[:, -1:]
#     configs = utils.get_nn_configs(2)
#     skf = StratifiedKFold(n_splits=10)
#     kfold_acc_mean = []
#     kfold_acc_se = []
#     kfold_auroc_mean = []
#     kfold_auroc_se = []
#     kfold_counter = 1
#     for train_index, test_index in skf.split(X, y):
#         print("Fold {}/10".format(kfold_counter))
#         models = Parallel(n_jobs=8, verbose=1)(
#             delayed(train_nn_parallely)(config, data_og, train_index, h1, h2) for config in configs)
#
#         global copy_models_tr
#         copy_models_tr = copy.deepcopy(models)
#         # models = [train_nn_parallely(config, X[train_index], y[train_index], h1, h2) for config in configs]
#
#         # bootstrap_results = []
#         # for b in range(5):
#         #     print("Bootstrap : {}/100".format(b + 1))
#         #     boot_test_data = resample(data_og[test_index], replace=True, stratify=y[test_index])
#         #     bootstrap_results.append(single_bootstrap(boot_test_data, models))
#
#         resampled_test_data = [resample(data_og[test_index], replace=True, stratify=y[test_index]) for _ in range(5)]
#         bootstrap_results = Parallel(n_jobs=8,require='sharedmem', verbose=1)(
#             delayed(single_bootstrap)(boot_test_data) for boot_test_data in resampled_test_data)
#
#         accuracies, aurocs = list(zip(*bootstrap_results))
#
#         sample_mean_acc = statistics.mean(accuracies)
#         standard_error_acc = math.sqrt(sum(list(map(lambda x: (x - sample_mean_acc) ** 2, accuracies))) / 99)
#
#         sample_mean_auroc = statistics.mean(aurocs)
#         standard_error_auroc = math.sqrt(sum(list(map(lambda x: (x - sample_mean_auroc) ** 2, aurocs))) / 99)
#
#         kfold_acc_mean.append(sample_mean_acc)
#         kfold_acc_se.append(standard_error_acc)
#
#         kfold_auroc_mean.append(sample_mean_auroc)
#         kfold_auroc_se.append(standard_error_auroc)
#         kfold_counter += 1
#         # trained_models = []
#
#     print("Concept {} Combination {}x{} Sample mean accuracy {} and Standard error accuracy {}"
#           .format(b, combinations[0], combinations[1], statistics.mean(kfold_acc_mean),
#                   statistics.mean(kfold_acc_se)))
#
#     print("Concept {} Combination {}x{} Sample mean auroc {} and standard error auroc {}".
#           format(b, combinations[0], combinations[1], statistics.mean(kfold_auroc_mean),
#                  statistics.mean(kfold_auroc_se)))
#
#     # return kfold_acc_mean, kfold_acc_se, kfold_auroc_mean, kfold_auroc_se
