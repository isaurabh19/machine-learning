def pca_reduction(X_T):
    rows, cols = X_T.shape
    sigma_matrix = np.cov(X_T.T)
    eigenvalues, eigenvectors = LA.eig(sigma_matrix)
    tot = sum(eigenvalues)
    var_exp = [(i / tot)*100 for i in eigenvalues]    
    cum_var_exp = np.cumsum(var_exp)
    columns_extract = []
    for i in range(len(cum_var_exp)):
        columns_extract.append(i)
        if cum_var_exp[i]>99:
            break
    pca_space = eigenvectors[:, columns_extract]
    X_T = np.matmul(pca_space.T, X_T.T).T
    return X_T
