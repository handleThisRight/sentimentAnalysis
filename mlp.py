import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import re
import statistics
import math
import csv
from sklearn.feature_extraction.text import *
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import confusion_matrix


data = np.array(pd.read_csv("./preprocessed/withPunc.csv"))
X = data[:,0]
Y = data[:,1]
Y= Y.astype(float)
X = X.astype(str)
print("check 1")



chargram = CountVectorizer(analyzer='char_wb', ngram_range=(3,6),max_df=0.5,min_df=0.01)
ngram = CountVectorizer(ngram_range=(1,4),max_df=0.98,min_df=0.001)
binary = CountVectorizer(binary=True)
tfid = TfidfVectorizer()
custom = DictVectorizer()


## checks for elongated words
def has_long(sentence):
	elong = re.compile("([a-zA-Z])\\1{2,}")
	flag = bool(elong.search(sentence))
	if(flag):
		return 1
	return 0

## list of negative words
file = pd.read_csv("negative-words.txt",sep='\n',header=None)
neg_words = list(np.array(file).reshape(len(file)),)


## builds custom features
def customFeatures(review):
	tokens = word_tokenize(review)
	allcaps = 0
	punctuation = 0
	elongated = has_long(review)
	negation = 0
	for word in tokens:
		if(word.isupper()):
			allcaps+=1/len(tokens)
		if(word in [',','!','?','..','...','....','"',"'"]):
			punctuation += 1/len(tokens)
		if(word in neg_words):
			negation += 1/len(tokens)
	pos= len(list(Counter([j for i,j in pos_tag(tokens)])))/len(tokens)
	d = {"allcaps":allcaps,"punctuation":punctuation,"elongated":elongated,"negation":negation}
	for t in tokens:
		d[t] = (d.get(t, 0) + 1)/len(tokens)
	return d


vectors = [custom, tfid, binary, chargram, ngram]
vname = ["custom", "tfid", "binary", "chargram", "ngram"]


print("check 2")
def accuracy(confusion_matrix):
   print("acc starts")
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

def accuracy(confusion_matrix):
  #It contains true positive, true negative, false positive and false negatives. Thus this matrix can be used to find accuracy, recall, etc. 
   diagonal_sum = confusion_matrix.trace() 
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements #formula for accuracy

def get_mlp(X_train, Y_train, X_val, Y_val):
  #This function inputs the reviews and labels for both test and train datasets. 
  #Then it trains with an MLP classifier with the specified hyperparameters (chosen after hit and trial as shown in report/readme/ppt)
  classifier = MLPClassifier(hidden_layer_sizes=(16,32,32), max_iter=50,activation = 'logistic',solver='adam', learning_rate_init = 0.0001) 
  classifier.fit(X_train, Y_train)
  y_pred = classifier.predict(X_val)#Then it predicts the labels of reviews of test set
  cm = confusion_matrix(y_pred, Y_val)
  print("Accuracy of MLPClassifier is ", accuracy(cm)) #Calls accuracy function defined above and gives accuracy of the predicted labels by comparing with actual testing labels.


for vector in vectors:
	X = vector.fit_transform(X)
	Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.30,random_state=1)
	get_mlp(Xtrain, Ytrain, Xtest, Ytest)
#calling the Mlp model
