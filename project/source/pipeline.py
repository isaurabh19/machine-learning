from source.data_generator import DataGenerator
from source.preprocessor import Preprocessor
from source.models.naive_bayes import NaiveBayes
from source.models.nn import NeuralNetworks
from source.models.dt import DecisionTree
from source.models.svm import SVMClassifier
from source.models.boosting import GradientBoosting
from source.models.rf import RandomForest
from source.models.lr import LogisticRegressionClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_scatter(data):
    plt.scatter(data[:, 0], data[:, 1], c=data[:, 2])
    plt.show()

def plot_features():
    pos, neg = data_generator.get_separate_datasets()
    pos_np = data_generator.convert_to_numpy_dataset(pos)
    neg_np = data_generator.convert_to_numpy_dataset(neg)
    for i in range(pos_np.shape[1]):
        sns.distplot(pos_np[:, i], hist=False, rug=False, kde=True)
        sns.distplot(neg_np[:, i], hist=False, rug=False, kde=True)
        plt.savefig("../data/top_n_feature_{}.png".format(i))
        plt.close()

np.random.seed(1)
data_generator = DataGenerator()
# plot_features()

dataset = data_generator.get_dataset()
numpy_dataset = data_generator.convert_to_numpy_dataset(dataset)
np.random.shuffle(numpy_dataset)

X = numpy_dataset[:, [0,1,5,7]]
y = numpy_dataset[:, -1:]
preprocessed_dataset = X
preprocessor = Preprocessor()
preprocessed_dataset = preprocessor.z_score(X)
# preprocessed_dataset = preprocessor.pca(X, 0.99)
data = np.concatenate((preprocessed_dataset, y), axis=1)

train, test = train_test_split(data, random_state=1, stratify=y)
# # lr = LogisticRegressionClassifier()
# # lr.run(train, test)
rf = RandomForest()
model = rf.run(train, test)
#
# gdb = GradientBoosting()
# gdb.run(train,test)
# svm = SVMClassifier()
# svm.run(train, test)

# dt = DecisionTree()
# dt.run(train, test)

# neural_network = NeuralNetworks()
# neural_network.train_nn(train, test)
# naive_bayes = NaiveBayes()
# naive_bayes.run(train, test)
