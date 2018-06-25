from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords

stops = set(stopwords.words("english")) 
deny_word=[u"aren't",u"couldn't",u"didn't",u"doesn't",u"don't",u"hadn't",u"haven't",u"hasn't",u"isn't",u"mightn't",u"mustn't",u"needn't",
                 u'not',u"shan't",u"shouldn't",u"wasn't",u"weren't",u"won't",u"wouldn't",u'no',u'nor']
degree_word=[u'too',u'very', u'further',u'so']
stops=stops-set(deny_word+degree_word)

wordnet_tags = ['n', 'v']

lemmatizer = WordNetLemmatizer()

def lemmatize(token, tag):
    if tag[0].lower() in ['n', 'v']:
        return lemmatizer.lemmatize(token, tag[0].lower())
    return token

def text_preprocess(words):
    words_soup = BeautifulSoup(words).get_text()
    
    words_soup = re.sub("n't", " not", words_soup)
    
    words_soup = re.sub("'", "", words_soup).lower()

    tagged_words = pos_tag(word_tokenize(words_soup))

    words_lemmatize_list=[lemmatize(token, tag) for token, tag in tagged_words]
    
    meaningful_words = [w for w in words_lemmatize_list if not w in stops]
    
    new_word=(" ".join(meaningful_words))
    
    return  re.sub("[^a-zA-Z]", " ", new_word) 

def text_preprocess2(words):
    words_soup = BeautifulSoup(words).get_text()
    
    words_soup = re.sub("n't", " not", words_soup)
    
    words_soup = re.sub("'", "", words_soup).lower()

    tagged_words = pos_tag(word_tokenize(words_soup))
    
    deny_flag=False
    
    words_lemmatize_list=[]
    
    for token, tag in tagged_words:
        if token in stops:
            continue
        if deny_flag and tag[0].lower() in ['v', 'j']:
            words_lemmatize_list.append('not'+lemmatize(token, tag))
        else:
            words_lemmatize_list.append(lemmatize(token, tag))
        if token=='not | no ':
            deny_flag=~deny_flag
        if token=='.' or token==',':
            deny_flag=False    
    new_word=(" ".join(words_lemmatize_list))
    
    return  re.sub("[^a-zA-Z]", " ", new_word) 


deny_list=['not','no' ,'never','nor','neither','none','nothing','nobody']
def text_preprocess3(words):
    words_soup = BeautifulSoup(words).get_text()
    
    words_soup = re.sub("n't", " not", words_soup)
    
    words_soup = re.sub("'", "", words_soup).lower()

    tagged_words = pos_tag(word_tokenize(words_soup))
    
    deny_flag=False
    
    words_lemmatize_list=[]
    
    for token, tag in tagged_words:
        if token in stops:
            continue
        if token in deny_list:
            deny_flag=~deny_flag
            continue
        if tag=='.' or tag==',' or tag==':':
            deny_flag=False  
            continue
        if deny_flag :
            words_lemmatize_list.append('not'+lemmatize(token, tag))
        else:
            words_lemmatize_list.append(lemmatize(token, tag))
    new_word=(" ".join(words_lemmatize_list))
    
    return  re.sub("[^a-zA-Z]", " ", new_word) 


deny_list=['not','no' ,'never','nor','neither','none','nothing','nobody']
def text_preprocess4(words):
    words_soup = BeautifulSoup(words).get_text()
    
    words_soup = re.sub("n't", " not", words_soup)
    
    words_soup = re.sub("'", "", words_soup).lower()

    tagged_words = pos_tag(word_tokenize(words_soup))
    
    deny_flag=False
    
    words_lemmatize_list=[]
    
    for token, tag in tagged_words:
        if token in stops:
            continue
        if token in deny_list:
            deny_flag=~deny_flag
            continue
        if tag=='.' or tag==',' or tag==':':
            deny_flag=False  
            continue
        if deny_flag and tag[0].lower() in ['v', 'j']:
            words_lemmatize_list.append('not'+lemmatize(token, tag))
        else:
            words_lemmatize_list.append(lemmatize(token, tag))
    new_word=(" ".join(words_lemmatize_list))
    
    return  re.sub("[^a-zA-Z]", " ", new_word) 



def get_score():
    output=pd.read_csv( "../submit/Bag_of_Words_model.csv")
    result=np.zeros(output.shape[0])
    output['true_value']=0
    for i in xrange(output.shape[0]):
        if int(output["id"][i].split("_")[1]) > 5:
            result[i]=1
        else:
            result[i]=0
    output['true_value']=result
    print np.mean(output['true_value']==output['sentiment'])