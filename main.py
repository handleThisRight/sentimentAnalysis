import pandas as pd 
import numpy as np

import csv
import pickle
#Vectorizers
from sklearn.feature_extraction.text import *
from sklearn.feature_extraction import DictVectorizer
# from sklearn.pipeline import FeatureUnion
# from zeugma.embeddings import EmbeddingTransformer


from sklearn.model_selection import train_test_split

#models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


import os


from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

from collections import Counter

import re


################################################################################
# data loading and shrinking
################################################################################

def loadData(path, colX, colY):
	data = np.array(pd.read_csv(path))

	X = data[:,colX]
	y = data[:,colY]

	y[y == 'rotten'] = 0
	y[y == 'fresh'] = 1
	y = y.astype(float)
	X = X.astype(str)
	# print(X.shape,y.shape)
	# u, count = np.unique(y, return_counts=True)
	# print(count)
	return X,y




Xa, ya = loadData("./Dataset/movieReviews.csv", 0, 1)

Xb, yb = loadData("./preprocessed/withPunc.csv", 0, 1)


#########################################################################################################
#########################################################################################################
# Vectorisation Techniques
# > Character-Ngrams
# > Ngrams
# > Binary
# > TF-IDF
# > Custom Freatures:
# 	> All capital
#	> Punctuation
#	> Elongated words
#	> Negation 
#########################################################################################################
#########################################################################################################


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




#Define Models
MNB = MultinomialNB()
BNB = BernoulliNB()
rf = RandomForestClassifier(random_state=0)
logReg = LogisticRegression(random_state=0)
svm = SVC(random_state = 42)
svm_linear = SVC(random_state = 42, kernel='linear')

models = [MNB, BNB, rf, logReg, svm, svm_linear]


########################################################################################################
########################################################################################################
# Pipeline:
# for every vectorisation method run every model
########################################################################################################
########################################################################################################

def vectorise(X, vectoriser, name):
	if(name=='custom'):
		X = [customFeatures(d) for d in X]
	vectorised = vectoriser.fit_transform(X)
	print(vectorised.shape)
	return vectorised


def runmodel(X, y, model):
	Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.30,random_state=1)
	model.fit(Xtrain, Ytrain)
	#save the model 
	pickle.dump(nn,open('Weights/model.sav','wb'))
	return model.score(Xtest,Ytest)




def pipeline(X, y):
	for i in range(len(vectors)):
		vector = vectors[i]
		name = vname[i]
		X_vectorised = vectorise(X, vector, name)
		for model in models:
			print(runmodel(X_vectorised, y, model))




print("Using raw dataset")
pipeline(Xa, ya)
print()

print("Using preprocessed data with punc")
pipeline(Xb, yb)
print()

