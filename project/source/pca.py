def pca_svd(X, d):
  # Data matrix X, X doesn't need to be 0-centered
    n, m = X.shape
  # Compute full SVD
    # It's not necessary to compute the full matrix of U or V
    U, Sigma, Vh = np.linalg.svd(X,full_matrices=False,
      compute_uv=True)
    # Transform X with SVD components
    U = U[:,[i for i in range(d)]]
    Sigma = Sigma[[i for i in range(d)]]
    X_svd = np.dot(U, np.diag(Sigma))
    return X_svd
