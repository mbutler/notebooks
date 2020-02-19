#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################
#before
#training time: 189.871 s
#prediction time: 19.153 s
#accuracy:  0.984072810011

#after dividing training set by 100
#training time: 0.099 s
#prediction time: 1.069 s
#accuracy:  0.884527872582

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.stats import itemfreq
clf = SVC(kernel="rbf", C=10000)
#the next two lines were added to cut the training to 1%
#it speeds everything up but reduces accuracy
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "accuracy: ", accuracy_score(pred, labels_test)
print "position: ", pred[42]
print "frequencies: ", itemfreq(pred)

#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data



#### store your predictions in a list named pred





