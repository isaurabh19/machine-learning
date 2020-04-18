from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np

"""
1. Split test data aside for final testing
1. Split into K folds
2. Build model architecture, class weights, regularization
3. train and predict
4. evaluate: precision, recall, f1, roc, accuracy: train and test
"""


class NeuralNetworks:
    def get_nn_model(self, no_dims):
        model = Sequential()
        model.add(Dense(2 * no_dims, input_dim=no_dims, activation="tanh"))
        model.add(Dense(no_dims, activation='relu'))
        model.add(Dense(2, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
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
        data = np.concatenate((train, test), axis=0)
        class_weights = compute_class_weight("balanced", np.unique(y_train), y_train.ravel())
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, restore_best_weights=True)
        model = self.get_nn_model(X_train.shape[1])
        model.fit(X_train, y_train, class_weight=class_weights, callbacks=[early_stopping], validation_split=0.1,
                  epochs=10000, verbose=1)
        y_scores = model.predict_proba(test[:, :-1].astype('int'))
        avergae_pr = average_precision_score(test[:, -1:].astype('int'), y_scores)
        print("Test area under pr curve {}".format(avergae_pr))
