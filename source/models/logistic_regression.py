import numpy as np
import pandas as pd
from numpy import linalg as LA
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold


class Logistic_Regression_model:
    def train(self, X):
        kf = StratifiedKFold(n_splits=10, shuffle=False)
        for train_index, test_index in kf.split(X[:, :-1], X[:, -1]):
            X_train, X_test = X[train_index, :], X[test_index, :]
            Y_train, Y_test = X_train[:, -1], X_test[:, -1]
            clf = LogisticRegression(random_state=0).fit(X_train, Y_train)
            Y_pred = clf.predict(Y_test)
            roc_scores[i].append(roc_auc_score(Y_test, Y_pred))
            accuracy_scores[i].append(accuracy_score(Y_test, Y_pred))
        return np.mean(accuracy_scores), np.mean(roc_scores)


