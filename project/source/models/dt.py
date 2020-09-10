from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import average_precision_score, roc_auc_score


class DecisionTree:

    def run(self, train, test):
        X_train = train[:, :-1]
        y_train = train[:, -1:]
        y_train = y_train.astype('int')
        X_test = test[:, :-1]
        y_test = test[:, -1:]
        y_test = y_test.astype('int')

        clf = DecisionTreeClassifier(criterion="gini")
        train_auroc = cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=10, n_jobs=8, verbose=1,
                                       error_score='raise')
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)[:, -1:]
        test_auroc = roc_auc_score(y_test, y_pred)

        print("Train auroc {}".format(train_auroc))
        print("Test auroc {}".format(test_auroc))
