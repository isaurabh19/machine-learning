from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.ensemble import GradientBoostingClassifier
from source.models.BaseModel import BaseClassifier
import statistics


class GradientBoosting(BaseClassifier):

    def run(self, train, test):
        X_train = train[:, :-1]
        y_train = train[:, -1:]
        y_train = y_train.astype('int')
        X_test = test[:, :-1]
        y_test = test[:, -1:]
        y_test = y_test.astype('int')

        clf = GradientBoostingClassifier(random_state=0)
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
        self.bootstrap(100, clf, test)
        return clf
