from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import compute_class_weight, resample
import statistics, math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class RandomForest:

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

        print("RF: Sample mean and standard error for Auroc {} {}".format(auroc_mean, auroc_se))
        print("RF: Sample mean and standard error for precision {} {}".format(precision_mean, precision_se))




    def predict_ranking(self, X_test, y_pred_scores, y_pred_labels, y_test):
        results = np.concatenate((X_test, y_pred_scores, np.array([y_pred_labels]).T), axis=1)
        dataset = np.concatenate((X_test, y_test), axis=1)

        positive_predicted = results[results[:, -1] == 1]
        true_positive = dataset[dataset[:, -1] == 1]
        for i in range(X_test.shape[1]):
            pos_feature = positive_predicted[:, i]
            true_label_feature = true_positive[:, i]
            sns.distplot(pos_feature, hist=False, rug=False, kde=True)
            sns.distplot(true_label_feature, hist=False, rug=False, kde=True)
            plt.savefig("../data/subset_feat_true_vs_predicted_pos_dist_{}.png".format(i))
            plt.close()
            print("Mean and SD for positive label feature {} are: {}, {}".format(i, np.mean(pos_feature),
                                                                                 np.std(pos_feature)))
            print("Mean and SD for true label feature {} are: {}, {}".format(i, np.mean(true_label_feature),
                                                                                 np.std(true_label_feature)))

    def run(self, train, test):
        X_train = train[:, :-1]
        y_train = train[:, -1:]
        y_train = y_train.astype('int')
        X_test = test[:, :-1]
        y_test = test[:, -1:]
        y_test = y_test.astype('int')

        class_weights = compute_class_weight("balanced", np.unique(y_train), y_train.ravel())
        clf = RandomForestClassifier(criterion="gini", class_weight={0: class_weights[0], 1: class_weights[1]})
        train_auroc = cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=10, n_jobs=8, verbose=1,
                                      error_score='raise')
        print(train_auroc)
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)[:, -1:]
        test_auroc = roc_auc_score(y_test, y_pred)

        print("Train auroc {}".format(statistics.mean(train_auroc)))
        print("Test auroc {}".format(test_auroc))

        y_pred_class = clf.predict(X_test)
        recall = recall_score(y_test, y_pred_class)
        precision = precision_score(y_test, y_pred_class)

        print("Recall {} and Precision {}".format(recall, precision))

        print("Feature importance {}".format(clf.feature_importances_))
        self.predict_ranking(X_test, y_pred, y_pred_class, y_test)
        self.bootstrap(100, clf, test)

        return clf
