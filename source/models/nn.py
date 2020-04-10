from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
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
    model.add(Dense(2 * no_dims, input_dim=no_dims, activation="tanh"))
    model.add(Dense(1, activation='tanh'))
    model.compile(optimizer='adadelta', loss='mean_squared_error')
    return model


def get_all_metrics(y_score, y_true, phase):
    pass

def get_best_model(scores_target_list):
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
        auroc = roc_auc_score(y_true, y_score)
        if auroc >= best_roc:
            best_roc = auroc
            best_model = model
        
    return best_model

def train_nn(train_validation_data, test_data, k=10):
    kf = KFold(k, random_state=10)
    class_weights = {0: 1.,
                     1: 2.1}
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, restore_best_weights=True)
    pred_target = []
    input_data = train_validation_data[:, :-1]
    targets = train_validation_data[:, -1:]
    for train_indices, test_indices in kf.split(input_data, targets):
        model = get_nn_model(input_data.shape[1])
        histoy = model.fit(input_data[train_indices], targets[train_indices], epochs=10000,
                           callbacks=[early_stopping], class_weight=class_weights, verbose=1)
        predictions_score = model.predict(input_data[test_indices], verbose=1)
        pred_target.append((predictions_score, targets[test_indices], model))

    best_model = get_best_model(pred_target)
    test_predictions = best_model.predict(test_data[:, :-1], verbose=1)
    get_all_metrics(test_predictions, test_data[:, -1:], 'test')



