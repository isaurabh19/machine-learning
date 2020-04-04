import numpy as np
from sklearn.decomposition import PCA
def pca_reduction(X, d):
	pca = PCA(n_components=d)
	newX = pca.fit_transform(X)
	return newX



