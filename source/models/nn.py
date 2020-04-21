from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.utils import compute_class_weight, resample
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np
from source.models.BaseModel import BaseClassifier
import math, statistics


class NeuralNetworks(BaseClassifier):
    def get_nn_model(self, no_dims):
        model = Sequential()
        model.add(Dense(no_dims, input_dim=no_dims, activation="tanh"))
        # model.add(Dense(int(no_dims/2), activation='relu'))
        model.add(Dense(2, activation='tanh'))
        model.add(Dense(1, activation='relu'))
        model.compile(optimizer='adadelta', loss='binary_crossentropy')
        return model

    def get_all_metrics(self, y_score, y_true, phase):
        pass

    def get_best_model(self, scores_target_list):
        """
        Find the model that has the max precision-recall auc or roc auc. Print training mean metrics
        :param scores_target_list:
        :return:
        """
        max_roc_auc = 0
        max_pr_auc = 0
        best_model = scores_target_list[0][2]
        best_roc = 0
        for y_score, y_true, model in scores_target_list:
            auroc = average_precision_score(y_true, y_score)
            if auroc >= best_roc:
                best_roc = auroc
                best_model = model

        return best_model

    def train_nn(self, train, test, k=10):
        # kf = KFold(k, random_state=10)
        X_train = train[:, :-1]
        y_train = train[:, -1:]
        y_train = y_train.astype('int')
        X_test = test[:, :-1]
        y_test = test[:, -1:]
        y_test = y_test.astype('int')

        data = np.concatenate((train, test), axis=0)
        class_weights = compute_class_weight("balanced", np.unique(y_train), y_train.ravel())
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)
        model = self.get_nn_model(X_train.shape[1])
        model.fit(X_train, y_train, class_weight=class_weights, validation_split=0.1,
                  epochs=200, verbose=1)
        y_scores = model.predict_proba(test[:, :-1].astype('int'))
        auroc = roc_auc_score(test[:, -1:].astype('int'), y_scores)
        y_labels = model.predict_classes(X_test)
        recall = recall_score(y_test, y_labels)
        precision = precision_score(y_test, y_labels)

        print("Test area under roc curve {}".format(auroc))
        print("NN: Recall {} and precision {}".format(recall, precision))

        self.bootstrap(100, model, test)


    def bootstrap(self, B, clf, test_data):
        aurocs = []
        precisions = []
        for b in range(B):
            resampled_data = resample(test_data, replace=True, stratify=test_data[:, -1:])
            X =  resampled_data[:, :-1]
            y = resampled_data[:, -1:]
            y = y.astype('int')
            y_score = clf.predict_proba(X)[:, -1:]
            y_class = clf.predict_classes(X)
            aurocs.append(roc_auc_score(y, y_score))
            precisions.append(precision_score(y, y_class))
        auroc_mean = statistics.mean(aurocs)
        auroc_se = math.sqrt(math.fabs(sum(list(map(lambda x: (x - auroc_mean) ** 2, aurocs))) / B-1))

        precision_mean = statistics.mean(precisions)
        precision_se = math.sqrt(math.fabs(sum(list(map(lambda x: (x - precision_mean) ** 2, precisions))) / B-1))

        print("RF: Sample mean and standard error for Auroc {} {}".format(auroc_mean, auroc_se))
        print("RF: Sample mean and standard error for precision {} {}".format(precision_mean, precision_se))