Sentiment Analysis

With the increased screen times during the pandemic, we largely depend on review sections to make choices while shopping, watching movies, making doctor appointments, and so on. We believe that the current review system has a few shortcomings and we wish to better it. This project aims to categorize user-written reviews on popular sites such as IMDb, Rotten Tomatoes, Amazon, etc, as “positive/negative”. We compared and analyzed the working and accuracy of different machine learning algorithms to classify reviews as positive/negative in an open-access review dataset. It was concluded that the Multi layer perceptron gave the best accuracy.


Run the following code in python to get all the necessary prerequisites to run the above code: 
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


Main.py:
Implements all the vectorisation techniques along with custom feature extraction.

textPrepropresor.py:
Does cleaning, normalisation and lemmatisation.

Plots consists of all the graphs

Weights consists of all the saved model.


