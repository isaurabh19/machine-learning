from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import compute_class_weight
import statistics
import numpy as np

class LogisticRegressionClassifier:

    def run(self, train, test):
        X_train = train[:, :-1]
        y_train = train[:, -1:]
        y_train = y_train.astype('int')
        X_test = test[:, :-1]
        y_test = test[:, -1:]
        y_test = y_test.astype('int')

        class_weights = compute_class_weight("balanced", np.unique(y_train), y_train.ravel())
        clf = LogisticRegression(random_state=0, class_weight={0:class_weights[0], 1:class_weights[1]})

        bg_clf = BaggingClassifier(clf, n_jobs=8, random_state=1, verbose=1)

        train_auroc = cross_val_score(bg_clf, X_train, y_train, scoring='roc_auc', cv=10, n_jobs=8, verbose=1,
                                      error_score='raise')
        print(train_auroc)
        bg_clf.fit(X_train, y_train)

        y_pred = bg_clf.predict_proba(X_test)[:, -1:]
        test_auroc = roc_auc_score(y_test, y_pred)

        print("LR: Train auroc {}".format(statistics.mean(train_auroc)))
        print("LR: Test auroc {}".format(test_auroc))

        y_pred_class = bg_clf.predict(X_test)
        recall = recall_score(y_test, y_pred_class)
        precision = precision_score(y_test, y_pred_class)

        print("LR: Recall {} and Precision {}".format(recall, precision))
