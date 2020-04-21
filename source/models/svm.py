from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils import compute_class_weight
from source.models.BaseModel import BaseClassifier
import statistics
import numpy as np


class SVMClassifier(BaseClassifier):

    def run(self, train, test):
        X_train = train[:, :-1]
        y_train = train[:, -1:]
        y_train = y_train.astype('int')
        X_test = test[:, :-1]
        y_test = test[:, -1:]
        y_test = y_test.astype('int')

        class_weights = compute_class_weight("balanced", np.unique(y_train), y_train.ravel())
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        scores = 'precision'

        clf = GridSearchCV(SVC(class_weight={0: class_weights[0], 1: class_weights[1]}), tuned_parameters,
                           scoring=scores, n_jobs=8, cv=10)
        clf.fit(X_train, y_train)
        print("Best model found at {}".format(clf.best_params_))

        best_model = clf.best_estimator_

        bg_clf = BaggingClassifier(best_model, n_jobs=8, random_state=1, verbose=1)

        train_auroc = cross_val_score(bg_clf, X_train, y_train, scoring='roc_auc', cv=10, n_jobs=8, verbose=1,
                                      error_score='raise')
        print(train_auroc)
        bg_clf.fit(X_train, y_train)

        y_pred = bg_clf.predict_proba(X_test)[:, -1:]
        test_auroc = roc_auc_score(y_test, y_pred)

        print("SVM: Train auroc {}".format(statistics.mean(train_auroc)))
        print("SVM: Test auroc {}".format(test_auroc))

        y_pred_class = bg_clf.predict(X_test)
        recall = recall_score(y_test, y_pred_class)
        precision = precision_score(y_test, y_pred_class)

        print("SVM: Recall {} and Precision {}".format(recall, precision))

        self.bootstrap(100, bg_clf, test)
