import pandas as pd 
import numpy as np

#nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

#spacy
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

#normalize
from normalise import normalise

import csv


import os


################################################################################
# data loading and shrinking
################################################################################

data = np.array(pd.read_csv("./Dataset/movieReviews.csv"))

X = data[:,1]
y = data[:,0]

y = y.astype(float)



u, count = np.unique(y, return_counts=True)
print("Initial count: [rotten fresh] --",count)

# shrinks the dataset keeping the class distribution uniform
# initial distribution 50:50 i.e. 240000 rotten + 240000 fresh
def shrink(X, y, size, random=True):
	ifresh = np.nonzero(y==0)
	irotten = np.nonzero(y==1)

	if(random==False):
		np.random.seed(0)
		np.random.shuffle(ifresh)
		np.random.shuffle(irotten)

	freshX = X[ifresh][:size]
	freshy = y[ifresh][:size]

	rottenX = X[irotten][:size]
	rotteny = y[irotten][:size]

	X_shrink = np.concatenate((freshX,rottenX))
	y_shrink = np.concatenate((freshy,rotteny))
	return X_shrink, y_shrink


# Uncomment this line to shrink the dataset
# X, y = shrink(X, y, 5)

u, count = np.unique(y, return_counts=True)
print("shrinked count: [rotten fresh] --",count)





################################################################################
# Tokenization
################################################################################


nltk_words = []
spacy_words = []
for review in X:

	nltk_words_per_review = word_tokenize(review)
	nltk_words.append(nltk_words_per_review)

	# doc = nlp(review)
	# spacy_words_per_review = [token.text for token in doc]
	# spacy_words.append(spacy_words_per_review)


#################################################################################
# Cleaning
#################################################################################


# punctuation removal - good choice for TF-IDF, Count, Binary vectorization
def removePunc(review):
	doc = nlp(review)
	tokens_without_punct_spacy = [t for t in doc if t.pos_ != 'PUNCT']
	return tokens_without_punct_spacy

# remove stop words 
file = pd.read_csv("negative-words.txt",sep='\n',header=None)
neg_words = list(np.array(file).reshape(len(file)),)
neg_words += ['not','no','none', 'nothing', 'neither', 'never', 'nobody', 'nowhere']

remove = ['<br','/>','>','<','/']

def removeStopWords(review, removeNeg=False):
	if(not removeNeg):
		customize_stop_words = neg_words
		for w in customize_stop_words:
			nlp.vocab[w].is_stop = False
	for w in remove:
		nlp.vocab[w].is_stop = True

	text_without_stop_words = [t for t in review if not t.is_stop]
	return text_without_stop_words


################################################################################
# Normalise
################################################################################

def norm(review):
	try:
		return ' '.join(normalise(review, variety="BrE", user_abbrevs={}, verbose=False))
	except:
		return review


################################################################################
# lemmatization
################################################################################

def nltk_lemmatize(review):
	nltk_lemm = []
	for tokens in nltk_words:
		wordnet_lemmatizer = WordNetLemmatizer()
		# vectorizing function to able to call on list of tokens
		lemmatize_words = np.vectorize(wordnet_lemmatizer.lemmatize)
		lemmatized_text = ' '.join(lemmatize_words(tokens))
		nltk_lemm.append(lemmatized_text)
	return nltk_lemm



def spacy_lemmatize(review):
	lemmas = [t.lemma_ for t in review]
	return ' '.join(lemmas)



#################################################################################
# pipeline :
# > normalise
# > remove punctuation (optional)
# > remove stop_words
# > lemmatize
#################################################################################

def preprocessing(filename, remPunc=False):
	with open(filename, 'w') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(["Review","Label"])
		for i in range(len(X)):
			if(i==12715):
				continue
			review = str(X[i])
			normalised = norm(review)
			doc = nlp(normalised)
			if(removePunc==True):
				doc = removePunc(doc)
			removedSW = removeStopWords(doc)
			lemmatized = spacy_lemmatize(removedSW)
			csvwriter.writerow([lemmatized,y[i]])
			print(i)



preprocessing('./preprocessed/withPunc.csv')
# preprocessing('./preprocessed/withoutPunc.csv', remPunc=True)