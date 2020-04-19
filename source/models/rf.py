from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import compute_class_weight
import statistics
import numpy as np


class RandomForest:

    def bootstrap(self, B, clf, test):

        for b in B:
            clf.predict_proba()

    def predict_ranking(self, clf):
        feature_importance = clf.feature_importances_
        path = clf.decision_path()

    def run(self, train, test):
        X_train = train[:, :-1]
        y_train = train[:, -1:]
        y_train = y_train.astype('int')
        X_test = test[:, :-1]
        y_test = test[:, -1:]
        y_test = y_test.astype('int')

        class_weights = compute_class_weight("balanced", np.unique(y_train), y_train.ravel())
        clf = RandomForestClassifier(criterion="gini", class_weight={0: class_weights[0], 1:class_weights[1]})
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


        print(clf.feature_importances_)
        indicator, n_nodes = clf.decision_path(X_test)

        return clf
