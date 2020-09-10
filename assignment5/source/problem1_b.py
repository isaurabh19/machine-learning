from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adadelta
from keras.regularizers import l2
import source.utils as ut
import numpy as np
import matplotlib.pyplot as plt
import statistics


# def run_parallely_no_bootstrap(h1, h2, data):
#     X = data[:, :-1]
#     y = data[:, -1:]
#     skf = StratifiedKFold(n_splits=10)
#     indices = [(train_index, test_index) for train_index, test_index in skf.split(X, y)]
#     results = Parallel(n_jobs=8, verbose=10)(
#         delayed(utils.single_fold_train)(data, index[0], index[1], h1, h2) for index in indices)
#     return results

def build_nn(h1, h2, ):
    model = Sequential()
    model.add(
        Dense(h1, input_dim=2, activation='tanh', kernel_regularizer=l2(0.01)))
    if h2 != 0:
        model.add(Dense(h2, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train(h1, h2, X, y, epochs=1000):
    model = build_nn(h1, h2)
    early_stopping = EarlyStopping(monitor='accuracy')
    model.fit(X, y, epochs=epochs, verbose=0)
    return model

step = 0.05
xx, yy, grid_data = ut.Utils().get_grid_data(step)

# grid_labels1 = np.array(list(map(utils.SquareConcept((-4., 3.), (2., 1.), (-2., -1.), 3).return_label, grid_data)))
# grid_labels2 = np.array(list(map(utils.SquareConcept((-4., 3.), (2., 0), (-1., -2.), 1).return_label, grid_data)))

grid_labels1 = ut.Utils().get_labels(ut.SquareConcept((-4., 3.), (2., 1.), (-2., -1.), 3), grid_data)
grid_labels2 = ut.Utils().get_labels(ut.SquareConcept((-4., 3.), (2., 0), (-1., -2.), 1), grid_data)

grid_labels = (grid_labels1, grid_labels2)

datasets = ut.Utils().get_dataset(1000)

for i, dataset in enumerate(datasets):
    for combinations in [(1, 0), (4, 0), (8, 0), (1, 3), (4, 3), (8, 3)]:
        # results = run_parallely(combinations[0], combinations[1], dataset)
        # acc, aurocs = list(zip(*results))
        #
        # print("Combination {}x{}:Train-validation accuracy {}".format(combinations[0], combinations[1],
        #                                                               statistics.mean(acc)))
        # print("Combination {}x{}:Train-validation auroc {}".format(combinations[0], combinations[1],
        #                                                            statistics.mean(aurocs)))

        # model = utils.train(combinations[0], combinations[1], dataset[:, :-1], dataset[:, -1:],800)
        model = build_nn(combinations[0], combinations[1])
        model.fit(dataset[:, :-1], dataset[:, -1:], epochs=1000)
        true_accuracy = model.evaluate(grid_data, grid_labels[i])[1]
        y_preds = model.predict(grid_data)
        auroc = roc_auc_score(grid_labels[i], y_preds)
        print("True accuracy for combination {}x{}: is {}".format(combinations[0], combinations[1], true_accuracy))
        print("True auroc for combination {}x{}: is {}".format(combinations[0], combinations[1], auroc))

        z = y_preds.reshape(xx.shape)
        z = np.flipud(z)
        plt.imshow(z, extent=[xx.min(), xx.max(), yy.min(), yy.max()])
        plt.show()
        # plt.savefig("{}_{}.png".format(combinations[0], combinations[1]))
