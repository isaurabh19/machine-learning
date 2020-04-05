from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

"""
1. Split test data aside for final testing
1. Split into K folds
2. Build model architecture, class weights, regularization
3. train and predict
4. evaluate: precision, recall, f1, roc, accuracy: train and test
"""


def get_nn_model(no_dims):
    model = Sequential()
    model.add(Dense(2 * no_dims, input(no_dims, ), activation="tanh"))
    model.add(Dense(1, activation='tanh'))
    model.compile(optimizer='adadelta', loss='mean_squared_error')
    return model


def train_nn(input_data, targets, k=10):
    kf = KFold(k, random_state=10)
    class_weights = {0: 1.,
                     1: 2.1}
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, restore_best_weights=True)
    pred_target=[]
    for train, test in kf.split(input_data, targets):
        model = get_nn_model(input_data.shape[1])
        histoy = model.fit(train[:, :-1], train[:, -1:], validation_data=test, epochs=10000, callbacks=early_stopping,
                           class_weight=class_weights, validation_freq=2, verbose=1)
        predictions = model.predict(test[:,:-1], verbose=1)
        pred_target.append((predictions, test[:, -1:]))

    return pred_target
