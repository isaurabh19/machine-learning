import source.utils as utils
from sklearn.metrics import roc_auc_score
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adadelta
from keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
import statistics


def build_nn(h1, h2, ):
    model = Sequential()
    model.add(
        Dense(h1, input_dim=2, activation='tanh'))
    if h2 != 0:
        model.add(Dense(h2, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train(h1, h2, X, y, epochs=1000):
    model = build_nn(h1, h2)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=3)
    model.fit(X, y, epochs=epochs, verbose=0)
    return model


datasets = utils.Utils().get_dataset(10000)
xx, yy, grid_data = utils.Utils().get_grid_data(0.05)

grid_data_label1 = utils.Utils().get_labels(utils.SquareConcept((-4., 3.), (2., 1.), (-2., -1.), 3), grid_data)
grid_data_label2 = utils.Utils().get_labels(utils.SquareConcept((-4., 3.), (2., 0), (-1., -2.), 1), grid_data)
grid_labels = (grid_data_label1, grid_data_label2)

nn_combinations = [(24, 9), (12, 3)]

for i, dataset in enumerate(datasets):
    for combination in nn_combinations:
        # results = utils.run_parallely(combination[0], combination[1], dataset, utils.single_fold_train)
        # acc, aurocs = list(zip(*results))
        #
        # print("Combination {}x{}:Train-validation accuracy {}".format(combination[0], combination[1],
        #                                                               statistics.mean(acc)))
        # print("Combination {}x{}:Train-validation auroc {}".format(combination[0], combination[1],
        #                                                            statistics.mean(aurocs)))
        model = build_nn(combination[0], combination[1])
        model.fit(dataset[:, :-1], dataset[:, -1:], verbose=0, epochs=1000)
        # model = train(combination[0], combination[1], dataset[:, :-1], dataset[:, -1:], epochs=1000)
        true_accuracy = model.evaluate(grid_data, grid_labels[i])[1]
        y_preds = model.predict(grid_data)
        auroc = roc_auc_score(grid_labels[i], y_preds)
        print("True accuracy for combination {}x{}: is {}".format(combination[0], combination[1], true_accuracy))
        print("True auroc for combination {}x{}: is {}".format(combination[0], combination[1], auroc))


        z = y_preds.reshape(xx.shape)
        z = np.flipud(z)
        plt.imshow(z, extent=[xx.min(), xx.max(), yy.min(), yy.max()])
        plt.show()
    print("Concept {} done".format(i + 1))
