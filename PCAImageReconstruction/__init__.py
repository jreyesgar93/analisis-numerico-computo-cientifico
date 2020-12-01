
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

class PCAImageReconstruction:
    
    def __init__(self):
    
        self.eigen_pairs = None
        self.variance = None
        self.required_dim = None
        self.keep_variance = None
        self.areapixels = None
        self.projection_matrix = None
        self.projection_data = None 
        self.eigenvalues = None
        self.eigenvectors = None
    
     
    def compress(self, X, normalize = False, keep_variance=0.99):
        self.areapixels = X.shape[1]
        if normalize == True:
            scaler = StandardScaler()
            X=scaler.fit_transform(X)
            
        cov_mat = np.cov(X.T)
        
        eigvals, eigvec= np.linalg.eig(cov_mat)
        
        self.eigenvalues = eigvals
        self.eigenvectors = eigvec
        
        eigen_pair = [(np.abs(eigvals[i]), eigvec[:,i]) for i in range(len(eigvec))]
        eigen_pair.sort(key=lambda x: x[0], reverse=True)

        
        required_variance = keep_variance * sum(eigvals)
        
        required_dim = 0
        variance = 0
        for i in range(len(eigen_pair)):
            variance += eigen_pair[i][0]
            if variance >= required_variance:
                required_dim = i + 1
                break


        
        self.eigen_pairs = eigen_pair
        self.variance = variance
        self.required_dim = required_dim
        self.keep_variance = keep_variance
        
        self.projection_matrix = np.empty(shape=(X.shape[1], self.required_dim))
        for index in range(required_dim):   
            self.projection_matrix[:, index] =self.eigen_pairs[index][1]
            
        self.projection_data = X.dot(self.projection_matrix)
                
    def plot_reconstructed_image(self,nimage):
        projected_image= np.expand_dims(self.projection_data[nimage], 0)
        plt.title('Reconstructed Image')
        plt.imshow(projected_image.dot(self.projection_matrix.T).reshape(int(np.sqrt(self.areapixels)),int(np.sqrt(self.areapixels))),"gray")
        return projected_image.dot(self.projection_matrix.T)
    
        
        
    
