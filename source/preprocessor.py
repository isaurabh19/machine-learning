from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Preprocessor:
    scalar = StandardScaler()

    def z_score(self, X):
        return self.scalar.fit_transform(X)

    def pca(self, X, retained_variance):
        pca = PCA(n_components=retained_variance)
        newX = pca.fit_transform(X)
        return newX
