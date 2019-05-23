#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import sys
import operator
from sklearn.model_selection import train_test_split
eps = np.finfo(float).eps
from numpy import log2 as log

def loadfile():
        filename='/home/neelesh/Downloads/Iris.csv'
        df = pd.read_csv(filename, header=None, names=(['A','B','C','D','Res']))
        X = df[['A','B','C','D']] 
        # X = df[['Work_accident','sales','promotion_last_5years','salary']] 
        Y = df[['Res']] 
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
        X_train=X_train.reset_index(drop=True)
        Y_train=Y_train.reset_index(drop=True)
        X_test=X_test.reset_index(drop=True)
        Y_test=Y_test.reset_index(drop=True)
        df=pd.concat([X_train,Y_train], axis=1)
        return df,X_train, X_test, Y_train, Y_test

# In[2]:


def getDistance(X1,X2):
	distance = 0
	for i in range(0,len(X2)):
                    distance += pow((X2[i] - X1[i]), 2)
	return math.sqrt(distance)


# In[3]:


def checkNeighbours(df,sample):
    Neighbour_dis= []
    for i in range(0,len(df)):
        dist = getDistance(df.iloc[i],sample)
        Neighbour_dis.append((df.iloc[i], dist))
    
    return Neighbour_dis
 


# In[4]:


def selectbestk(myneighbour,k):
    myneighbour.sort(key=operator.itemgetter(1))
    bestk=[]
    for i in range(0,k):
        bestk.append(myneighbour[i][0])
    return bestk
    


# In[5]:


def predict( kneighbours,k):


        classVotes = {}
        for x in range(len(kneighbours)):
            response = kneighbours[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]

    


# In[ ]:





# In[6]:


def getAccuracy(testSet, predictions):
        correct = 0
        TP={}
        Pred={}
        Real={}
        for x in range(len(testSet)):
            if(testSet[x] in Real):
                Real[testSet[x]]+=1
            else:
                Real[testSet[x]]=1
            if(predictions[x] in Pred):
                Pred[predictions[x]]+=1
            else:
                Pred[predictions[x]]=1
            if testSet[x] == predictions[x]:
                if(testSet[x] in TP):
                    TP[testSet[x]]+=1
                else:
                    TP[testSet[x]]=1
                correct += 1
        return TP,Pred,Real,correct


# In[7]:


def show_result():
        df,X_train, X_test, Y_train, Y_test=loadfile()
        res=[]
        k=3
        TP={}
        Pred={}
        Real={}
        if(len(sys.argv)>1):
            Test_Filename=sys.argv[1]
            testdf =pd.read_csv(Test_Filename, header=None, names=(['A','B','C','D','Res']))
            X_test=testdf[['A','B','C','D']]  
            Y_test=testdf[['Res']]

        for i in range(0,len(X_test)):
            test=X_test.iloc[i]
            neighbours = checkNeighbours(df,test)
            kneighbours=selectbestk(neighbours,k)
            p=predict(kneighbours,test)
            res.append(p)            
        test_array=np.array(Y_test['Res'])
        attrib=Y_test['Res'].unique()

        TP,Pred,Real,correct=getAccuracy(test_array,res)

        print "Accuracy: ",(correct*100.0)/len(test_array),"%"
        print "========================================"
        for val in attrib:

            print "Details of attribute",val
            
            print"Precision: ",float(TP[val]*100)/Pred[val],"%"
            print"Recall: ",float(TP[val]*100)/Real[val],"%"
            print "========================================"
#             print("F1 score: ", f1_score)

show_result()


# In[ ]:





# In[ ]:





# In[ ]:




