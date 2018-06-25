import pandas as pd     
import nltk
import re
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.corpus import stopwords # Import the stop word list
import os
import gc
import lightgbm as lgb
from utils import *
from tqdm import tqdm  
train_path='../data/processed_train8.tsv'
test_path='../data/processed_test8.tsv'

if not os.path.exists(train_path):
    train = pd.read_csv("../data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    train['review_new']=train['review'].apply(text_preprocess4)
    train[['id','sentiment','review_new']].to_csv(train_path,index=False,sep='$')
    del train
    gc.collect()
    
if not os.path.exists(test_path):
    test= pd.read_csv("../data/testData.tsv", header=0, delimiter="\t", quoting=3)
    test['review_new']=test['review'].apply(text_preprocess4)
    test[['id','review_new']].to_csv(test_path,index=False,sep='$')
    del test
    gc.collect()
    
alltrain=pd.read_csv(train_path,sep='$')
test=pd.read_csv(test_path,sep='$')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             ngram_range=(1,3),
                             max_features = 200000) 
train_data_features = vectorizer.fit_transform(alltrain['review_new'].values)
test_data_features = vectorizer.transform(test['review_new'].values)
train_data_features_new=train_data_features.copy()
test_data_features_new=test_data_features.copy()

from sklearn import preprocessing
binarizer = preprocessing.Binarizer()
train_data_features_new=binarizer.transform(train_data_features_new)
test_data_features_new=binarizer.transform(test_data_features_new)


filter_1=(alltrain['sentiment'].values==1)
A=np.sum(train_data_features_new[filter_1],axis=0)
B=np.sum(train_data_features_new[~filter_1],axis=0)
A=np.array(A,dtype=float)
B=np.array(B,dtype=float)
A=A/A.sum()
B=B/B.sum()
r=(A-B)/(A+B)
r=np.abs(r)

train_data_features_new=train_data_features_new.multiply(r)
test_data_features_new=test_data_features_new.multiply(r)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='l2', tol=0.0001, 
                         C=1, fit_intercept=True, intercept_scaling=1.0, 
                         class_weight=None, random_state=2018)


model.fit(train_data_features_new,alltrain['sentiment'].values) 
result_lr1=model.predict(test_data_features_new)
result_lr2=model.predict_proba(test_data_features_new)[:,1] 
output = pd.DataFrame( data={"id":test["id"], "sentiment":result_lr1} )
output.to_csv( "../submit/Bag_of_Words_model.csv", index=False, quoting=3 )
get_score()