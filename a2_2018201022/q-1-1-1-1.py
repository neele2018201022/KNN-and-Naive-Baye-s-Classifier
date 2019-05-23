#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import operator
import sys
from sklearn.model_selection import train_test_split
eps = np.finfo(float).eps
from numpy import log2 as log

def loaddb(filename):
            df = pd.read_csv(filename,sep=" ", header=None, names=(['Res','A','B','C','D','E','F','G']))
            df=df[['A','B','C','D','E','F','Res']].reset_index(drop=True)
            X= df[['A','B','C','D','E','F']] 
            Y= df[['Res']] 
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

    


# In[6]:


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
    


# In[7]:


print "<===============Result for Robot1====================>"
def show_result(filename):
    df,X_train, X_test, Y_train, Y_test=loaddb(filename)
    res=[]
    k=3
    TP={}
    Pred={}
    Real={}
    if(len(sys.argv)>1):
				Test_Filename=sys.argv[1]
				testdf = pd.read_csv(Test_Filename,sep=" ", header=None, names=(['Res','A','B','C','D','E','F','G']))
				X_test=testdf[['A','B','C','D','E','F']] 
				Y_test=testdf[['Res']]

    for i in range(0,len(X_test)):
        test=X_test.iloc[i]
        neighbours = checkNeighbours(df,test)
        kneighbours=selectbestk(neighbours,k)
        p=predict(kneighbours,test)
        res.append(p)            
    test_array=np.array(Y_test['Res'])
    attrib=Y_test['Res'].unique()

    accuracy, precision, recall,f1_score = calculate_recall_precision(test_array,res)
    print "Accuracy: {0}%".format(accuracy)
    print "Precision: {0}%".format(precision)
    print "Recall: {0}%".format(recall)
    print "F1 score: {0}%".format(f1_score)
show_result('/home/neelesh/Downloads/RobotDataset/Robot1')
print "-----------------------------------------------------"
print "<===============Result for Robot2====================>"
show_result('/home/neelesh/Downloads/RobotDataset/Robot2')


# In[ ]:





# In[ ]:





# In[ ]:




