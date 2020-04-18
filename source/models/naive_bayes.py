from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import average_precision_score, roc_auc_score


class NaiveBayes:
    def train_nb(self, train_validation_data, test_data, k=10):
        pass

    def run(self, train, test):
        X_train = train[:, :-1]
        y_train = train[:, -1:]
        y_train = y_train.astype('int')
        X_test = test[:, :-1]
        y_test = test[:, -1:]
        y_test = y_test.astype('int')
        gnb = GaussianNB()
        train_auc_pr = cross_val_score(gnb, X_train, y_train, scoring='average_precision', cv=10, n_jobs=8, verbose=1,
                                       error_score='raise')
        gnb.fit(X_train, y_train.ravel())
        y_pred_labels = gnb.predict(X_test)
        y_pred_scores = gnb.predict_proba(X_test)[:, -1:]
        test_auc_pr = average_precision_score(y_test, y_pred_scores)
        auroc = roc_auc_score(y_test, y_pred_scores)

        print("Train auc pr {}".format(train_auc_pr))
        print("Test auc pr {}".format(test_auc_pr))
        print("Test auroc {}".format(auroc))
