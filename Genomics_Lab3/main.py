import pandas as pd
import statsmodels.api as sm 
import numpy as np
import statsmodels

class Lab3(object):
    
    def create_data(self,snp_lines) :
        '''
        Input - the snp_lines parsed at the beginning of the notebook
        Output - You should return the 53 x 3902 dataframe
        '''
        #start code here
        changedData = {}
        for line in snp_lines:
            data = line.split("\t")
            chromozome = data[0]
            position = data[1]
            dataKey = str(chromozome)+":"+str(position)
            row = np.zeros(53)
            for i,v in enumerate(data[9:]):
                reference = v[0]
                alternate = v[2]
                if reference == '.' or alternate == '.':
                    row[i] = np.nan
                else:
                    row[i] = int(reference)+int(alternate)
            changedData[dataKey] = row
        return pd.DataFrame(changedData)
    
        #end code here

    def create_target(self,header_line) :
        '''
        Input - the header_line parsed at the beginning of the notebook
        Output - a list of values(either 0 or 1)
        '''
        #start code here
        data = header_line.split("\t")
        values = []
        for i in data:
            if "yellow" in i:
                values.append(1)
            if "dark" in i:
                values.append(0)
        return values
        
        #end code here
    
    def logistic_reg_per_snp(self,df) :
        '''
        Input - snp_data dataframe
        Output - list of pvalues and list of betavalues
        '''
        #start code here
        pValues = []
        bValues= []
        for i in df:
            if i =='target':
                continue
            obj = sm.Logit(df["target"],sm.add_constant(df[i]), missing='drop').fit(method='bfgs', disp=False)
            pValues.append(round(obj.pvalues[1], 9))
            bValues.append(round(obj.params[1],5))
        return pValues, bValues
        
        #end code here
    
    
    def get_top_snps(self,snp_data,p_values) :
        '''
        Input - snp dataframe with target column and p_values calculated previously
        Output - list of 5 tuples, each with chromosome and position
        '''
        #start code here
        columns =  snp_data.columns[np.argpartition(p_values, 5)[:5]]
        returnVal = []
        for i in columns:
            data = i.split(":")
            tu = (data[0], data[1])
            returnVal.append(tu)
        return returnVal
        
        #end code here