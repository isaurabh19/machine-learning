from source.data_generator import DataGenerator
from source.preprocessor import Preprocessor
from source.models.naive_bayes import NaiveBayes
from source.models.nn import NeuralNetworks
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


data_generator = DataGenerator()
# plot_features()

dataset = data_generator.get_dataset()
numpy_dataset = data_generator.convert_to_numpy_dataset(dataset)

X = numpy_dataset[:, :-2]
y = numpy_dataset[:, -1:]
preprocessor = Preprocessor()
preprocessed_dataset = preprocessor.z_score(X)
preprocessed_dataset = preprocessor.pca(X, 0.99)
data = np.concatenate((preprocessed_dataset, y), axis=1)

train, test = train_test_split(data, random_state=1, stratify=y)
nerual_network = NeuralNetworks()
nerual_network.train_nn(train, test)
# naive_bayes = NaiveBayes()
# naive_bayes.run(train, test)
