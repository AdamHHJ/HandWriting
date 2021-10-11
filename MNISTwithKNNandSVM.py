#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


def load_data():
    #load training and test data
    with open("train-images.idx3-ubyte") as rf:
        trainX = np.fromfile(file=rf,dtype=np.uint8)
    with open("train-labels.idx1-ubyte") as rf:
        trainY = np.fromfile(file=rf,dtype=np.uint8)
    with open("t10k-images.idx3-ubyte") as rf:
        testX = np.fromfile(file=rf,dtype=np.uint8)
    with open("t10k-labels.idx1-ubyte") as rf:
        testY = np.fromfile(file=rf,dtype=np.uint8)
    return (trainX[16:].reshape((-1,784)).astype(np.float),trainY[8:].reshape((-1)).astype(np.float),
           testX[16:].reshape((-1,784)).astype(np.float),testY[8:].reshape((-1)).astype(np.float))


# In[3]:


trainX,trainY,testX,testY = load_data()
print("Original training data size = ",trainX.shape)
print("Original training label size = ",trainY.shape)
print("Original test data size = ",testX.shape)
print("Original test label size = ",testY.shape)
plt.title("A example of digit 5")
plt.imshow(trainX[0].reshape(28,28),cmap="gray")
plt.show()

trainX,_,trainY,_ = train_test_split(trainX,trainY,train_size=0.1,random_state=1,stratify=trainY)
_,testX,_,testY = train_test_split(testX,testY,test_size=0.1,random_state=1,stratify=testY)
print("Sampled training data size = ",trainX.shape)
print("Sampled training label size = ",trainY.shape)
print("Sampled test data size = ",testX.shape)
print("Sampled test label size = ",testY.shape)

# In[4]:


#Training a kNN classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(trainX,trainY)
train_acc = knn_classifier.score(trainX,trainY)
test_acc = knn_classifier.score(testX,testY)
print("The training accuracy of kNN is {:.2f}%".format(train_acc*100))
print("The test accuracy of kNN is {:.2f}%".format(test_acc*100))


# In[5]:


#Training a SVM classifier
svm_classifier = SVC(random_state=1)
svm_classifier.fit(trainX,trainY)
train_acc = svm_classifier.score(trainX,trainY)
test_acc = svm_classifier.score(testX,testY)
print("The training accuracy of SVM is {:.2f}%".format(train_acc*100))
print("The test accuracy of SVM is {:.2f}%".format(test_acc*100))

