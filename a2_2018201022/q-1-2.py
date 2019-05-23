#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
import math
import sys
from sklearn.model_selection import train_test_split
eps = np.finfo(float).eps
from numpy import log2 as log
def loadfile():
        df =pd.read_csv('/home/neelesh/Downloads/data.csv',header=None, names=(['A','B','C','D','E','F','G','H','I','Res','K','L','M','N']))
        X=df[['B','C','D','E','F','G','H','I','K','L','M','N']]
        Y=df[['Res']]   
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state =1)
        X_train=X_train.reset_index(drop=True)
        Y_train=Y_train.reset_index(drop=True)
        X_test=X_test.reset_index(drop=True)
        Y_test=Y_test.reset_index(drop=True)
        df=pd.concat([X_train,Y_train], axis=1) 
        testSet=pd.concat([X_test,Y_test], axis=1)
        return df,X_train, X_test, Y_train, Y_test,testSet


# In[17]:


def getmean(arr):
        return sum(arr)*1.0/(len(arr))
def getstdev(arr):
        avg = sum(arr)*1.0/float(len(arr))
        variance = sum([pow(x-avg,2) for x in arr])/(float(len(arr)-1)+eps)
        return math.sqrt(variance)
def helperfunc_mean(df,temp):
        return [(getmean(df[attribute]), getstdev(df[attribute])) for attribute in list(temp)]    


# In[18]:


def getmeandev(df):
        att={}
        info={}
        attValue = np.unique(df['Res'])
        for value in attValue:
                att[value]=[]
                att[value]=df[df['Res'] == value].reset_index(drop=True)
        temp=df.drop(['Res'], axis=1)
        attValue1 = list(temp)
        for value in attValue:
                info[value]=[]
                info[value] =helperfunc_mean(att[value],temp)
        return info    


# In[19]:


def calcGaussianProb(df,meandev,test):
        res= {}
        attValue = np.unique(df['Res'])
        for value in attValue:
                res[value]=1
        for value in attValue:
                for i in range(0,len(meandev[value])):
                        mean, stdev = meandev[value][i]
                        exponent = math.exp(-(math.pow(test[i]-mean,2)/(2*math.pow(stdev,2)+eps)))
                        res[value] *= (1/(eps+(math.sqrt(2*math.pi)*stdev)))*exponent
        return res


# In[20]:


def predict(df,meandev,test):
        res= calcGaussianProb(df,meandev,test)
        ans=None
        attValue = np.unique(df['Res'])
        for value in attValue:
            if (ans is None or res[value]>prob):
                prob=res[value]
                ans=value
        return ans
    
def help_predictor(df,meandev, testSet):
        res= []
        for i in range(len(testSet)):
                    result = predict(df,meandev, testSet.iloc[i])
                    res.append(result)
        return res


# In[21]:


def calculate_recall_precision(original,res):
        TP=0
        FP=0
        TN= 0
        FN= 0
        f1_score=0
        for i in range(0, len(original)):

                if res[i] == 1:
                    if res[i] == original[i]:
                        TP+= 1
                    else:
                        FP+= 1
                else:
                    if res[i] == original[i]:
                        TN+= 1
                    else:
                        FN+= 1

        precision=0
        recall=0
        if(TP!=0 or TN!=0):
                accuracy = (TP+TN)*1.0/(TP + TN +FP +FN)
        if(TP!=0):
                precision = TP*1.0/(TP + FP)
                recall = TP*1.0/(TP + FN)
                f1_score = 2 / ((1 / precision) + (1 / recall))
        print "True +ve=",TP,"True -ve=",TN,"False +ve=",FP,"False -ve=",FN                    

        return accuracy*100, precision*100, recall*100,f1_score*100    


# In[22]:


def show():
        df,X_train, X_test, Y_train, Y_test,testSet=loadfile()
        meandev= getmeandev(df)
        if(len(sys.argv)>1):
            Test_Filename=sys.argv[1]
            testdf =pd.read_csv(Test_Filename,header=None, names=(['A','B','C','D','E','F','G','H','I','Res','K','L','M','N']))
            X_test=testdf[['B','C','D','E','F','G','H','I','K','L','M','N']] 
            Y_test=testdf[['Res']]
            testSet=pd.concat([X_test,Y_test], axis=1)
        
        predictions = help_predictor(df,meandev, testSet)
        testarr=np.array(testSet['Res'])
        a,p,r,f = calculate_recall_precision(testarr, predictions)
        print('Accuracy: {0}%'.format(a))
        print('Precison: {0}%'.format(p))
        print('Recall: {0}%'.format(r))
        print('F1 score: {0}%'.format(f))

show()


# In[ ]:





# In[ ]:





# In[ ]:




