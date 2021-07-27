import numpy as np
import scipy.spatial.distance as dist
from scipy import stats
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time

class Question1(object):
    def bayesClassifier(self,data,pi,means,cov):
        # print(data.shape)
        # print(means.shape)
        # print(cov.shape)
        cInv = np.linalg.pinv(cov)
        firstTerm = np.log(pi)
        secondTerm = np.dot(np.dot(means,cInv),np.transpose(data))
        thirdTerm = 0.5*np.dot(np.dot(means,cInv),np.transpose(means))
        # algorithm based on lecture notes( Lecture 3)
        objectFunct = np.transpose(firstTerm)+np.transpose(secondTerm)-np.diag(thirdTerm)
        return np.argmax(objectFunct, axis=1)

    def classifierError(self,truelabels,estimatedlabels):
        incorrectPred = np.absolute(estimatedlabels - truelabels)
        totalErrors = np.sum(incorrectPred>0)
        return totalErrors/estimatedlabels.size


class Question2(object):
    def trainLDA(self,trainfeat,trainlabel):
        nlabels = int(trainlabel.max())+1 # Assuming all labels up to nlabels exist.
        pi = np.zeros(nlabels)            # Store your prior in here
        means = np.zeros((nlabels,trainfeat.shape[1]))            # Store the class means in here
        cov = np.zeros((trainfeat.shape[1],trainfeat.shape[1]))   # Store the covariance matrix in here
        # Put your code below
        covSum = 0
        # slogorithm based on lecture 3
        for label in range(nlabels):
            data = trainfeat[trainlabel==label]
            pi[label] = trainfeat[trainlabel==label].size/trainfeat.size
            means[label] = np.mean(trainfeat[trainlabel==label],axis = 0)
            covSum+= np.dot(np.transpose(data-means[label]),(data-means[label]))
        totalDataSize = ((trainfeat.size)/2)-nlabels
        cov=covSum/totalDataSize
        return (pi,means,cov)

    def estTrainingLabelsAndError(self,trainingdata,traininglabels):
        q1 = Question1()
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)
        lpi, lmeans, lcov = self.trainLDA(trainingdata,traininglabels)
        esttrlabels = q1.bayesClassifier(trainingdata, lpi, lmeans, lcov)
        trerror = q1.classifierError(traininglabels, esttrlabels) 
        # Don't change the output!
        return (esttrlabels, trerror)

    def estValidationLabelsAndError(self,trainingdata,traininglabels,valdata,vallabels):
        q1 = Question1()
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)
        lpi, lmeans, lcov = self.trainLDA(trainingdata,traininglabels)
        estvallabels = q1.bayesClassifier(valdata, lpi, lmeans, lcov)
        valerror = q1.classifierError(vallabels, estvallabels) 
        # Don't change the output!
        return (estvallabels, valerror)


class Question3(object):
    def kNN(self,trainfeat,trainlabel,testfeat, k):
         # create k values for each testfeat
        labels = np.zeros(testfeat.shape[0])
        for i in range(testfeat.shape[0]):
            feature = np.array([testfeat[i]])
            # distance for each testfeat from train feat 
            distance = np.argpartition(dist.cdist(trainfeat, feature), k, 0)
            topKelements = []
            # get top k element for each test feature 
            for j in range(0, k):
                topKelements.append(trainlabel[distance[j]])
            # chose the most common one
            labels[i] = stats.mode(topKelements)[0][0]
        return labels
       

    def kNN_errors(self,trainingdata, traininglabels, valdata, vallabels):
        q1 = Question1()
        trainingError = np.zeros(4)
        validationError = np.zeros(4)
        k_array = [1,3,4,5]
        for i in range(len(k_array)):
            # Please store the two error arrays in increasing order with k
            # This function should call your previous self.kNN() function.
            # Put your code below
            trainLabelsTest = self.kNN(trainingdata, traininglabels, trainingdata, k_array[i])
            trainErrors = q1.classifierError(trainLabelsTest, traininglabels)
            valLabelsTest = self.kNN(trainingdata, traininglabels, valdata, k_array[i])
            valErrors = q1.classifierError(valLabelsTest, vallabels)
            trainingError[i] = trainErrors
            validationError[i] = valErrors
        # Don't change the output!
        return (trainingError, validationError)

class Question4(object):
    def sklearn_kNN(self,traindata,trainlabels,valdata,vallabels):
        q1 = Question1()
        classifier, valerror, fitTime, predTime = (None, None, None, None)
        
        classifier = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm= "brute",metric = "euclidean")
        start = time.time()
        classifier.fit(traindata, trainlabels)
        fitTime = time.time()-start
        start = time.time()
        valerror = q1.classifierError(vallabels, classifier.predict(valdata))
        predTime = time.time()-start

        # Don't change the output!
        return (classifier, valerror, fitTime, predTime)

    def sklearn_LDA(self,traindata,trainlabels,valdata,vallabels):
        classifier, valerror, fitTime, predTime = (None, None, None, None)
        
        q1 = Question1()
        classifier = LinearDiscriminantAnalysis()
        start = time.time()
        classifier.fit(traindata, trainlabels)
        fitTime = time.time()-start
        start = time.time()
        valerror = q1.classifierError(vallabels, classifier.predict(valdata))
        predTime = time.time()-start
        # Don't change the output!
        return (classifier, valerror, fitTime, predTime)

###
