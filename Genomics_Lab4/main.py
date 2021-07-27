import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
class Lab4(object):
    
    def expectation_maximization(self,read_mapping,tr_lengths,n_iterations) :
        #start code here
        # reads = len(read_mapping)
        pValues = np.full(30,1/30)
        answer = np.zeros((30, n_iterations+1))
        answer[:,0] = pValues
        tr_lengths = np.array(tr_lengths)
        for i in range(1,n_iterations+1):
            zValues = np.zeros((len(read_mapping), 30))
            for j,r in enumerate(read_mapping):
                for k in r:
                    zValues[j,k] = pValues[k]/np.sum(pValues[r])
            theta = np.mean(zValues, axis=0)/tr_lengths
            pValues = theta/np.sum(theta)
            answer[:,i] = pValues
        return list(answer)
        #end code here

    def prepare_data(self,lines_genes) :
        '''
        Input - list of strings where each string corresponds to expression levels of a gene across 3005 cells
        Output - gene expression dataframe
        '''
        #start code here
        genes = []
        for i in lines_genes:
            i = i.strip()
            genes.append(i.split(" "))
        data = pd.DataFrame(genes)
        data = data.astype(float)
        data+=1
        data = np.log(data)
        data = data.round(5)
        data = data.T
        columns = []
        for i in range(data.shape[1]):
            columns.append("Gene_"+str(i))
        data.columns = columns
        return data
        
        #end code here
    
    def identify_less_expressive_genes(self,df) :
        '''
        Input - gene expression dataframe
        Output - list of column names which are expressed in less than 25 cells
        '''
        #start code here
        x = np.count_nonzero(df, axis=0)<25
        answer = []
        for i in range(len(x)):
            if x[i] == True:
                answer.append(df.columns[i])
        return answer
        
        #end code here
    
    
    def perform_pca(self,df) :
        '''
        Input - df_new
        Output - numpy array containing the top 50 principal components of the data.
        '''
        #start code here
        model = PCA(n_components = 50, random_state=365)
        model = model.fit_transform(df.to_numpy())
        return np.round(model,5)
        
        #end code here
    
    def perform_tsne(self,pca_data) :
        '''
        Input - pca_data
        Output - numpy array containing the top 2 tsne components of the data.
        '''
        #start code here
        
        #end code here