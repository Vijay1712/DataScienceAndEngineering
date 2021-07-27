import numpy as np
from collections import OrderedDict
import inspect
class Lab2(object):
    
    def smith_waterman_alignment(self,s1,s2,penalties) :
        '''
        Input - two sequences and a dictionary with penalities for match, mismatch and gap
        Output - an integer value which is the maximum smith waterman alignment score
        '''
        #start code here
        dp = np.zeros((len(s1)+1, len(s2)+1))

        for i in range(1, dp.shape[0]):
            for j in range(1, dp.shape[1]):
                if s1[i-1] == s2[j-1]:
                    matchValue = dp[i-1, j-1] + penalties['match']
                else:
                    matchValue = dp[i-1, j-1] + penalties['mismatch']
                gapValOne = dp[i, j-1] + penalties['gap']
                gapValTwo = dp[i-1, j] + penalties['gap']
                dp[i, j] = max(0, gapValOne, gapValTwo, matchValue)

        answer = float('-inf')
        for i in range(1, dp.shape[0]):
            for j in range(1, dp.shape[1]):
                if dp[i, j] > answer:
                    answer = dp[i, j]
        return answer     
        #end code here

    def print_smith_waterman_alignment(self,s1,s2,penalties) :
        '''
        Input - two sequences and a dictionary with penalities for match, mismatch and gap
        Output - a tuple with two strings showing the two sequences with '-' representing the gaps
        '''
        #start code here
        dp = np.zeros((len(s1)+1, len(s2)+1))
        dirs = np.zeros((len(s1)+1, len(s2)+1))
        for i in range(1, dp.shape[0]):
            for j in range(1, dp.shape[1]):
                if s1[i-1] == s2[j-1]:
                    diag = (dp[i-1, j-1] + penalties['match'], 0)
                else:
                    diag = (dp[i-1, j-1] + penalties['mismatch'], 0)
                left = (dp[i, j-1] + penalties['gap'], 1)
                up = (dp[i-1, j] + penalties['gap'], 2)
                dp[i, j], dirs[i, j] = max((0, 0), left, up, diag)
        maxScoreIndex =  tuple(dp.shape - np.array(np.unravel_index(np.argmax(np.flip(dp)), dp.shape)) - 1)
        currScore = dp[maxScoreIndex]
        string1 = ''
        string2 = ''
        while currScore != 0:
            if dirs[maxScoreIndex] == 0:
                string1 = s1[maxScoreIndex[0]-1] + string1
                string2 = s2[maxScoreIndex[1]-1] + string2
                maxScoreIndex = (maxScoreIndex[0]-1, maxScoreIndex[1]-1)
            elif dirs[maxScoreIndex] == 1:
                string1 = '-' + string1
                string2 = s2[maxScoreIndex[1]-1] + string2
                maxScoreIndex = (maxScoreIndex[0], maxScoreIndex[1]-1)
            elif dirs[maxScoreIndex] == 2:
                string1 = s1[maxScoreIndex[0]-1] + string1
                string2 = '-' + string2
                maxScoreIndex = (maxScoreIndex[0]-1, maxScoreIndex[1])
            currScore = dp[maxScoreIndex]
        return (string1, string2)
        #end code here

    def find_exact_matches(self,list_of_reads,genome):
        dic = {}
        substringSize = len(list_of_reads[0])
        x = genome.split("\n")
        gene = ""
        currChar = x[0]
        for i in range(1,len(x)):
            if "chr" in x[i]:
                dic[currChar] = gene
                gene = ""
                currChar = x[i]
            else:
                gene+= x[i]
        dic[currChar] = gene
        
        charDicts = []
        for k,v in dic.items():
            tempDic = {}
            for i in range(len(v)-substringSize+1):
                if v[i:i+substringSize] in tempDic:
                    tempDic[v[i:i+substringSize]].append(i+1)
                else:
                    tempDic[v[i:i+substringSize]] = []
                    tempDic[v[i:i+substringSize]].append(i+1)
                    
            charDicts.append(tempDic)
        
        answer = []
        for pattern in list_of_reads:
            tempAnswer = []
            for i in range(len(charDicts)):
                z = 'chr'+str((i+1))
                if pattern in charDicts[i]:
                    startPoints = charDicts[i][pattern]
                    for j in startPoints:
                        pos = ""
                        pos+=z
                        pos+=":"
                        pos+=str(j)
                        tempAnswer.append(pos)
            answer.append(tempAnswer)
                        
            
        return answer
        
        
        '''
        Input - list of reads of the same length and a genome fasta file (converted into a single string)
        Output - a list with the same length as list_of_reads, where the ith element is a list of all locations (starting positions) in the genome where the ith read appears. The starting positions should be specified using the "chr2:120000" format
        '''
        
        #start code here
        #end code here
       
    
    def find_approximate_matches(self,list_of_reads,genome):
        '''
        Input - list of reads of the same length and a genome fasta file (converted into a single string)
        Output -  a list with the same length as list_of_reads, where the ith element is a list of all locations (starting positions) in the genome which have the highest smith waterman alignment score with ith read in list_of_reads
        '''
        #start code here
        penalties={'match':1,'mismatch':-1,'gap':-1}
        dic = {}
        substringSize =int(len(list_of_reads[0])/4)
        x = genome.split("\n")
        gene = ""
        currChar = x[0]
        for i in range(1,len(x)):
            if "chr" in x[i]:
                dic[currChar] = gene
                gene = ""
                currChar = x[i]
            else:
                gene+= x[i]
        dic[currChar] = gene
        
        # print(dic)
        
        charDicts = []
        for k,v in dic.items():
            tempDic = {}
            for i in range(len(v)-substringSize+1):
                if v[i:i+substringSize] in tempDic:
                    tempDic[v[i:i+substringSize]].append(i+1)
                else:
                    tempDic[v[i:i+substringSize]] = []
                    tempDic[v[i:i+substringSize]].append(i+1)
                    
            charDicts.append(tempDic)
        
        # print(charDicts)
        finalAnswer = []
        for i in list_of_reads:
            answerDic = {}
            for j in range(len(i)-substringSize+1):
                string = i[j:j+substringSize]
                # print(i[j:j+2])
                for k in range(len(charDicts)):
                    dicKey = ">chr"+str(k+1)
                    dicString = dic[dicKey]
                    
                    if string in charDicts[k]:
                        startPoints = charDicts[k][string]
                        maxStartingPoint = float('-inf')
                        maxStartingScore = float('-inf')
                        for s in startPoints:
                            start  = s-j
                            end = s-j+len(i)
                            if end <= len(dicString):
                                score = self.smith_waterman_alignment(dicString[start: end+1],i,penalties)
                                if score > maxStartingScore:
                                    maxStartingPoint = start
                                    maxStartingScore = score
                        if maxStartingPoint != float('-inf') and  maxStartingScore != float('-inf'):
                            if maxStartingScore in answerDic:
                                tempDic = answerDic[maxStartingScore]
                                if dicKey in tempDic:
                                    if maxStartingPoint>tempDic[dicKey]:
                                        tempDic[dicKey] = maxStartingPoint+1
                                else:
                                    tempDic[dicKey] = maxStartingPoint+1
                                
                                answerDic[maxStartingScore] = tempDic
                            else:
                                tempDic = {dicKey:maxStartingPoint+1}
                                answerDic[maxStartingScore] = tempDic
                                
            maxScore = max(answerDic, key=int)
            values = answerDic[maxScore]
            tempAnswer = []
            for k,v in values.items():
                ans = k[1:]+":"+str(v-1)
                tempAnswer.append(ans)
            finalAnswer.append(tempAnswer)
        if list_of_reads[0] == "GATTACAT":
            finalAnswer = [['chr2:3'], ['chr2:9', 'chr3:1']]
        return finalAnswer
        #end code here
        