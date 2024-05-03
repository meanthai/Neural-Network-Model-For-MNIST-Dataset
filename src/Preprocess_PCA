# PCA step by step explaination by Fredtou
def PCA(X):
    num_components = 0
    
    #Step-1: Apply normalization method
    # Scaling data using Z-score normalization
    scaler = StandardScaler()
    X_meaned = scaler.fit_transform(X)
    
    #Step-2: Creating covariance matrix
    cov_mat = np.cov(X_meaned, rowvar = False)
     
    #Step-3: Calculating eigen values and eigen vectors
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Step-4: Sorting the eigen vectors in descending order based on the eigen values
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    total = sum(eigen_values)
    var_exp = [( i /total ) * 100 for i in sorted_eigenvalue]
    cum_var_exp = np.cumsum(var_exp)
    for ite, percentage in enumerate(cum_var_exp):
        if percentage >= 95: # Take the features that make the variance percentage over 95%
            num_components = ite
            print(ite)
            break
    # print("percentage of cummulative variance per eigenvector in order: ", cum_var_exp)
         
    #Step-5: Extracting the final dataset after applying dimensionality reduction
    eigenvector_subset = sorted_eigenvectors[:, : num_components]
     
    #Step-6:Transforming the processed matrix
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
    return X_reduced
