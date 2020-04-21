from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, precision_score
import math, statistics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class BaseClassifier:

    def predict_ranking(self, X_test, y_pred_scores, y_pred_labels, y_test):
        results = np.concatenate((X_test, y_pred_scores, np.array([y_pred_labels]).T), axis=1)
        dataset = np.concatenate((X_test, y_test), axis=1)

        positive_predicted = results[results[:, -1] == 1]
        true_positive = dataset[dataset[:, -1] == 1]
        for i in range(X_test.shape[1]):
            pos_feature = positive_predicted[:, i]
            neg_feature = true_positive[:, i]
            sns.distplot(pos_feature, hist=False, rug=False, kde=True)
            sns.distplot(neg_feature, hist=False, rug=False, kde=True)
            plt.savefig("../data/true_vs_predicted_pos_dist_{}.png".format(i))
            plt.close()
            print("Mean and SD for positive label feature {} are: {}, {}".format(i, np.mean(pos_feature),
                                                                                 np.std(pos_feature)))
            print("Mean and SD for negative label feature {} are: {}, {}".format(i, np.mean(neg_feature),
                                                                                 np.std(neg_feature)))

    def bootstrap(self, B, clf, test_data):
        aurocs = []
        precisions = []
        for b in range(B):
            resampled_data = resample(test_data, replace=True, stratify=test_data[:, -1:])
            X =  resampled_data[:, :-1]
            y = resampled_data[:, -1:]
            y = y.astype('int')
            y_score = clf.predict_proba(X)[:, -1:]
            y_class = clf.predict(X)
            aurocs.append(roc_auc_score(y, y_score))
            precisions.append(precision_score(y, y_class))
        auroc_mean = statistics.mean(aurocs)
        auroc_se = math.sqrt(math.fabs(sum(list(map(lambda x: (x - auroc_mean) ** 2, aurocs))) / B-1))

        precision_mean = statistics.mean(precisions)
        precision_se = math.sqrt(math.fabs(sum(list(map(lambda x: (x - precision_mean) ** 2, precisions))) / B-1))

        print("Sample mean and standard error for Auroc {} {}".format(auroc_mean, auroc_se))
        print("Sample mean and standard error for precision {} {}".format(precision_mean, precision_se))