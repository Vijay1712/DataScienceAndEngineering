import numpy as np
from sklearn import neighbors
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.decomposition import PCA
from PIL import Image
from sklearn.cluster import KMeans
import scipy.spatial.distance as dist
from matplotlib.colors import ListedColormap

class Question1(object):
    def pcaeig(self,data):
        """ Implement PCA via the eigendecomposition.
       
        Parameters:
        1. data     (N, d) numpy ndarray. Each row as a feature vector.

        Outputs:
        1. W        (d,d) numpy array. PCA transformation matrix (Note that each row of the matrix should be a principal component)
        2. s        (d,) numpy array. Vector consisting of the amount  of variance explained in the data by each PCA feature. 
        Note that the PCA features are ordered in decreasing amount of variance explained, by convention.
        """
        covMatrix = np.matmul(data.T, data) / data.shape[0]
        eigenVals, eigenVecs = np.linalg.eigh(covMatrix)
        s = np.flip(eigenVals)
        W = np.fliplr(eigenVecs)
    
        # Remember to check your data types
        return (W.T,s)
    
    def pcadimreduce(self,data, W, k):
        """ Implements dimension reduction via PCA.
        
        Parameters:
        1. data     (N, d) numpy ndarray. Each row as a feature vector.
        2. W        (d,d) numpy array. PCA transformation matrix
        3. k        number. Number of PCA features to retain
        
        Outputs:
        1. reduced_data  (N,k) numpy ndarray, where each row contains PCA features corresponding to its input feature.
        """       
        x = W[:k]
        reduced_data = np.dot(x,data.T).T         
        
        return reduced_data
    
    def pcareconstruct(self,pcadata, W, k):
        """ Implements dimension reduction via PCA.
        
        Parameters:
        1. pcadata     (N, k) numpy ndarray. Each row as a PCA vector. (e.g. generated from pcadimreduce)
        2. W        (d,d) numpy array. PCA transformation matrix
        3. k        number. Number of PCA features 
        
        Outputs:
        1. reconstructed_data  (N,d) numpy ndarray, where the i-th row contains the reconstruction of the original i-th input feature vector (in `data`) based on the PCA features contained in `pcadata`.
        """
        x = W[:k]
        reconstructed_data = np.dot(x.T, pcadata.T).T
        
        return reconstructed_data
    
    def pcasvd(self, data): 
        """Implements PCA via SVD.
        
        Parameters: 
        1. data     (N, d) numpy ndarray. Each row as a feature vector.
        
        Returns: 
        1. Wsvd     (d,d) numpy array. PCA transformation matrix (Note that each row of the matrix should be a principal component)
        2. ssvd       (d,) numpy array. Vector consisting of the amount  of variance explained in the data by each PCA feature. 
        Note that the PCA features are ordered in decreasing amount of variance explained, by convention.
        """
        SVD = np.linalg.svd(data)
        Wsvd = SVD[2]
        ssvd =pow(SVD[1],2)/data.shape[0]
        
        return Wsvd, ssvd
        

class Question2(object):
    
    def unexp_var(self, X):
        """Returns an numpy array with the fraction of unexplained variance on X by retaining the first k principal components for k =1,...200.
        Parameters:
        1. X        The input image
        
        Returns:
        1. pca      The PCA object fit on X 
        2. unexpv   A (200,) numpy ndarray, where the i-th element contains the percentage of unexplained variance on X by retaining i+1 principal components
        """
        unexpv=np.zeros(200)
        for k in range(1,201):
            pca = PCA(n_components=k)
            pca.fit(X)
            unexpv[k-1]=(1-sum(pca.explained_variance_ratio_))
        
        return pca,unexpv
    
    def pca_approx(self, X_t, pca, i):
        """Returns an approimation of `X_t` using the the first `i`  principal components (learned from `X`).
        
        Parameters:
            1. X_t      The input image to be approximated
            2. pca      The PCA object to use for the transform
            3. i        Number of principal components to retain
            
        Returns: 
            1. recon_img    The reconstructed approximation of X_t using the first i principal components learned from X (As a sanity check it should be of size (1,4096))
        """
        X_t = X_t.reshape(1,-1)
        w = pca.transform(X_t)
        for j in range(len(w)):
            for k in range(len(w[j])):
                if k>i:
                    w[j][k] = 0
        recon_img = pca.inverse_transform(w)
        
        return recon_img
    
   

class Question3(object):
    
    def pca_classify(self, traindata,trainlabels, valdata, vallabels):
        """Returns validation errors using 1-NN on the PCA features using 1,2,...,256 PCA features, the minimum validation error, and number of PCA features used.
        
        Parameters: 
            traindata       
            trainlabels
            valdata
            valabels
            
        Returns:
            ve      numpy array of length 256 containing the validation errors using 1,...,256 features
            min_ve  minimum validation error
            pca_feat Number of PCA features to retain. Integer.
        """
        ve = np.zeros(256)
        for i in range(256):
            classifier = PCA(n_components = i+1)
            classifier.fit(traindata)
            trainReduced = classifier.transform(traindata)
            valReduced = classifier.transform(valdata)
            classifier_1 = neighbors.KNeighborsClassifier(n_neighbors = 1)
            classifier_1.fit(trainReduced, trainlabels)
            
            ve[i] = 1 - classifier_1.score(valReduced, vallabels)
            
        min_ve = np.min(ve)
        pca_feat = np.argmin(ve)+1

        
        return ve, min_ve, pca_feat
    
