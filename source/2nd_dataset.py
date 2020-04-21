import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from keras import models
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score,roc_curve, precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Reshape, ZeroPadding2D, Dropout, Flatten, Dense, Activation
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

def keras_network_results(X_train, y_train, X_test, y_test):
    inputs = keras.Input(shape=(X_train.shape[1],), name='signal')
    x = layers.Dense(8, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(4, activation='relu', name='dense_1')(inputs)
    outputs = layers.Dense(2, name='predictions')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')
    model.summary()
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer=keras.optimizers.RMSprop())
    history = model.fit(np.array(X_train), np.array(y_train),batch_size=1,epochs=100)
    predictions = model.predict(np.array(X_test))
    my_predictions = [np.argmax(i) for i in predictions]
    print(np.unique(my_predictions))
    return my_predictions

## Cross validation
def cross_validation(X, B, model):
    kf = StratifiedKFold(n_splits=2, shuffle=True)
    roc_scores = [[] for i in range(100)]
    accuracy_scores = [[] for i in range(100)]
    precisions = [[] for i in range(100)]

    for train_index, test_index in kf.split(X[:, :-1], X[:, -1]):
        X_train, X_test = X[train_index, :], X[test_index, :]
        model.fit(X_train[:,:-1], X_train[:,-1])
        for i in range(B):
            X_test = resample(X_test, n_samples=5000, replace=True)
            X_t = X_test[:, :-1]
            Y_t = X_test[:, -1]
            y_values = model.predict(X_t)
            roc_scores[i].append(roc_auc_score(Y_t, y_values))
            accuracy_scores[i].append(accuracy_score(Y_t, y_values))
            precisions[i].append(precision_score(Y_t, y_values, average='weighted'))

    final_accuracy = []
    final_auc = []
    final_precision = []
    for i in accuracy_scores:
        final_accuracy.append(np.mean(i))
    for i in roc_scores:
        final_auc.append(np.mean(i))
    for i in precisions:
        final_precision.append(np.mean(i))
    return np.mean(final_accuracy), np.std(final_accuracy), np.mean(final_auc), np.std(final_auc),\
           np.mean(final_precision), np.std(final_precision)



if __name__=="__main__":
    df = pd.read_csv('../data/songs_complete_data.csv', index_col=0)
    ##plotting features and their dependence

    ## Dropping URI
    df.drop(['URI'], axis=1, inplace=True)

    ## Dropping Title
    df.drop(['Title'], axis=1, inplace=True)

    ## Dropping lyrics
    df.drop(['lyrics'], axis=1, inplace=True)

    ## take a copy of the present dataframe
    df_final = df.copy()

    ## Categorical data preprocess
    df_final['Artist'] = df_final['Artist'].astype('category')
    df_final['Artist'] = df_final['Artist'].cat.codes
    df_final['Genre'] = df_final['Genre'].astype('category')
    df_final['Genre'] = df_final['Genre'].cat.codes

    ##plot features
    pos_np = np.asarray(df_final.loc[df['Top100'] == 1])
    neg_np = np.asarray(df_final.loc[df['Top100'] == 0])
    for i in range(pos_np.shape[1]):
        print(i)
        sns.distplot(pos_np[:, i], hist=False, rug=False, kde=True)
        sns.distplot(neg_np[:, i], hist=False, rug=False, kde=True)
        plt.savefig("top_n_feature_{}.png".format(df_final.columns[i]))
        plt.close()

    imp_cols = [2,3,5,8,9,15,16]
    non_imp_cols = []
    for i in range(len(df_final.columns)):
        print(i)
        print(df_final.columns[i])
        non_imp_cols.append(i)


    ## Setting target columns
    df_finalY = df_final['Top100']
    df_finalX = df_final.drop(['Top100'], axis=1, inplace=False)

    ##Scaling the data
    scaler = preprocessing.StandardScaler().fit(df_finalX)
    df_finalX = scaler.transform(df_finalX)

    ## with full features
    X_train, X_test, y_train, y_test = train_test_split(df_finalX, df_finalY, test_size=0.20, random_state=42)
    y_pred_val = keras_network_results(X_train, y_train, X_test, y_test)
    print("Accuracy : ", accuracy_score(y_test, y_pred_val))
    print("AUC : ", roc_auc_score(y_test, y_pred_val))
    print("precision macro : ", precision_score(y_test, y_pred_val, average='macro'))
    print("precision micro : ", precision_score(y_test, y_pred_val, average='micro'))
    print("precision weighted : ", precision_score(y_test, y_pred_val, average='weighted'))

    print("with complete dataset")
    print("SVM results bootstrapping")
    clf = SVC(gamma='auto')
    bootstrapping_results = cross_validation(np.column_stack((df_finalX, df_finalY)), 100, clf)
    print(bootstrapping_results)

    print("Random Forest bootstrapping")
    clf = RandomForestClassifier(max_depth=40, random_state=42)
    bootstrapping_results = cross_validation(np.column_stack((df_finalX, df_finalY)), 100, clf)
    print(bootstrapping_results)

    print("Logistic Regression bootstrapping")
    clf = LogisticRegression(random_state=42)
    bootstrapping_results = cross_validation(np.column_stack((df_finalX, df_finalY)), 100, clf)
    print(bootstrapping_results)

    # Reducing Dimensions
    df_final.drop(["Artist"], axis=1, inplace=True)
    df_final.drop(["Key"], axis=1, inplace=True)
    df_final.drop(["Mode"], axis=1, inplace=True)
    df_final.drop(["Speechiness"], axis=1, inplace=True)
    df_final.drop(["Liveness"], axis=1, inplace=True)
    df_final.drop(["Valence"], axis=1, inplace=True)
    df_final.drop(["Tempo"], axis=1, inplace=True)
    df_final.drop(["Duration"], axis=1, inplace=True)
    df_final.drop(["Time_Signature"], axis=1, inplace=True)
    print("remaining columns are : ", df_final.columns)

    ## Setting target columns
    df_finalY = df_final['Top100']
    df_finalX = df_final.drop(['Top100'], axis=1, inplace=False)

    ## with reduced dataset
    print("with reduced dataset")
    X_train, X_test, y_train, y_test = train_test_split(df_finalX, df_finalY, test_size=0.20, random_state=42)
    y_pred_val = keras_network_results(X_train, y_train, X_test, y_test)
    print("Accuracy : ", accuracy_score(y_test, y_pred_val))
    print("AUC : ", roc_auc_score(y_test, y_pred_val))
    print("precision macro : ", precision_score(y_test, y_pred_val, average='macro'))
    print("precision micro : ", precision_score(y_test, y_pred_val, average='micro'))
    print("precision weighted : ", precision_score(y_test, y_pred_val, average='weighted'))


    print("SVM results bootstrapping")
    clf = SVC(gamma='auto')
    bootstrapping_results = cross_validation(np.column_stack((df_finalX, df_finalY)), 100, clf)
    print(bootstrapping_results)

    print("Random Forest bootstrapping")
    clf = RandomForestClassifier(max_depth=40, random_state=42)
    bootstrapping_results = cross_validation(np.column_stack((df_finalX, df_finalY)), 100, clf)
    print(bootstrapping_results)

    print("Logistic Regression bootstrapping")
    clf = LogisticRegression(random_state=42)
    bootstrapping_results = cross_validation(np.column_stack((df_finalX, df_finalY)), 100, clf)
    print(bootstrapping_results)

