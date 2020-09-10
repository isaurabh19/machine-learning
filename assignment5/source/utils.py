from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2, l1
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import numpy as np
import keras
import itertools


class SquareConcept:
    square1 = square2 = square3 = tuple()
    length = 0

    def __init__(self, square1, square2, square3, length):
        self.square1 = square1
        self.square2 = square2
        self.square3 = square3
        self.length = length

    def _check_inside_square(self, square, point):
        if (square[0] <= point[0] <= square[0] + self.length) and (square[1] >= point[1] >= square[1] - self.length):
            return True
        return False

    def return_label(self, point):
        if self._check_inside_square(self.square1, point) or self._check_inside_square(self.square2,
                                                                                       point) or self._check_inside_square(
            self.square3, point):
            return 1
        return 0


class Utils:
    def get_labelled_data(self, concept, data):
        labels = np.array(list(map(concept.return_label, data)))
        return np.concatenate((data, np.array([labels]).T), axis=1)

    def get_labels(self, concept, data):
        labels = np.array(list(map(concept.return_label, data)))
        return labels.T

    def get_grid_data(self, step):
        xx, yy = np.meshgrid(np.arange(-6, 6 + step, step), np.arange(-4, 4 + step, step))
        grid_data = np.c_[xx.ravel(), yy.ravel()]
        return xx, yy, grid_data

    def get_dataset(self, sample_size):
        concept1 = SquareConcept((-4., 3.), (2., 1.), (-2., -1.), 3)
        concept2 = SquareConcept((-4., 3.), (2., 0), (-1., -2.), 1)
        np.random.seed(10)
        data = np.random.uniform(low=(-6.0, -4.0), high=(6.0, 4.0), size=(sample_size, 2))
        labelled_data1 = self.get_labelled_data(concept1, data)
        labelled_data2 = self.get_labelled_data(concept2, data)
        return labelled_data1, labelled_data2

    def run_parallely(self, h1, h2, data, func):
        X = data[:, :-1]
        y = data[:, -1:]
        skf = StratifiedKFold(n_splits=10)
        indices = [(train_index, test_index) for train_index, test_index in skf.split(X, y)]
        results = Parallel(n_jobs=8, verbose=10)(
            delayed(func)(data, index[0], index[1], h1, h2) for index in indices)
        return results

    def single_fold_train(self, data, train_index, test_index, h1, h2):
        # model = utils.build_nn(h1, h2)
        # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3)
        # model.fit(X[train_index], y[train_index], callbacks=[early_stopping], epochs=500, verbose=0)
        X = data[:, :-1]
        y = data[:, -1:]
        model = self.train(h1, h2, X[train_index], y[train_index])
        accuracy = model.evaluate(X[test_index], y[test_index])[1]
        y_pred = model.predict(X[test_index])
        auroc = roc_auc_score(y[test_index], y_pred)

        return accuracy, auroc

    def build_nn(self, h1, h2, activation='tanh', activation_op='sigmoid', loss='binary_crossentropy',
                 kernel_initiliazer='glorot_uniform'):
        model = Sequential()
        model.add(
            Dense(h1, input_dim=2, activation=activation, kernel_initializer=kernel_initiliazer))
        if h2 != 0:
            model.add(Dense(h2, activation=activation, kernel_initializer=kernel_initiliazer))
        model.add(Dense(1, activation=activation_op))
        opt = Adadelta()
        model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        return model

    def train(self, h1, h2, X, y, epochs=500):
        model = self.build_nn(h1, h2)
        early_stopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=3)
        model.fit(X, y, callbacks=[early_stopping], epochs=epochs, verbose=0)
        return model

    def get_nn_configs(self, n=10):
        np.random.seed(10)
        loss = ['mean_squared_error', 'binary_crossentropy']
        initializers = ['glorot_uniform', 'zeros', 'RandomUniform', 'glorot_normal', 'lecun_normal', 'he_uniform']

        configs = [i for i in itertools.product(initializers, loss)]
        indices = np.random.choice(np.arange(len(loss) * len(initializers)), n)
        return [configs[index] for index in indices]
