from source.data_generator import DataGenerator
from source.preprocessor import Preprocessor

data_generator = DataGenerator()
dataset = data_generator.get_dataset()
numpy_dataset = data_generator.convert_to_numpy_dataset(dataset)
X = numpy_dataset[:, :-1]
y = numpy_dataset[:, -1:]
preprocessor = Preprocessor()
preprocessed_dataset = preprocessor.z_score(X)
preprocessed_dataset = preprocessor.pca(X, 0.95)

print(preprocessed_dataset)