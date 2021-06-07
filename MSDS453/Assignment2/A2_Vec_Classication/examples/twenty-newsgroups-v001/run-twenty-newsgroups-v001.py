# Text Classification Example with Selected Newsgroups from Twenty Newsgroups

# Author: Thomas W. Miller (2019-03-08)

# Compares text classification performance under random forests
# Six vectorization methods compared:
#     TfidfVectorizer from Scikit Learn
#     CountVectorizer from Scikit Learn
#     HashingVectorizer from Scikit Learn
#     Doc2Vec from gensim (dimension 50)
#     Doc2Vec from gensim (dimension 100)
#     Doc2Vec from gensim (dimension 200)

# See example data and code from 
# https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html

# The 20 newsgroups dataset comprises around 18000 newsgroups 
# posts on 20 topics split in two subsets: one for training (or development) 
# and the other one for testing (or for performance evaluation). 
# The split between the train and test set is based upon messages 
# posted before and after a specific date.

###############################################################################
### Note. Install all required packages prior to importing
###############################################################################
import multiprocessing

import re,string
from pprint import pprint

import numpy as np

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import TfidfVectorizer,\
    CountVectorizer, HashingVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import nltk
stoplist = nltk.corpus.stopwords.words('english')
DROP_STOPWORDS = False

from nltk.stem import PorterStemmer
#Functionality to turn stemming on or off
STEMMING = False  # judgment call, parsed documents more readable if False

MAX_NGRAM_LENGTH = 1  # try 1 for unigrams... 2 for bigrams... and so on
VECTOR_LENGTH = 1000  # set vector length for TF-IDF and Doc2Vec
WRITE_VECTORS_TO_FILE = False
SET_RANDOM = 9999

# subsets of newsgroups may be selected
# SELECT_CATEGORY = 'COMPUTERS'
# SELECT_CATEGORY = 'RECREATION'
# SELECT_CATEGORY = 'SCIENCE'
# SELECT_CATEGORY = 'TALK'
SELECT_CATEGORY = 'ALL'

##############################
### Utility Functions 
##############################
# define list of codes to be dropped from document
# carriage-returns, line-feeds, tabs
codelist = ['\r', '\n', '\t']    

# text parsing function for entire document string
def parse_doc(text):
    text = text.lower()
    text = re.sub(r'&(.)+', "", text)  # no & references  
    text = re.sub(r'pct', 'percent', text)  # replace pct abreviation  
    text = re.sub(r"[^\w\d'\s]+", '', text)  # no punct except single quote 
    text = re.sub(r'[^\x00-\x7f]',r'', text)  # no non-ASCII strings    
    if text.isdigit(): text = ""  # omit words that are all digits    
    for code in codelist:
        text = re.sub(code, ' ', text)  # get rid of escape codes  
    # replace multiple spacess with one space
    text = re.sub('\s+', ' ', text)        
    return text

# text parsing for words within entire document string
# splits the document string into words/tokens
# parses the words and then recreates a document string
# returns list of parsed words/tokens and parsed document string
def parse_words(text): 
    # split document into individual words
    tokens=text.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out tokens that are one or two characters long
    tokens = [word for word in tokens if len(word) > 2]
    # filter out tokens that are more than twenty characters long
    tokens = [word for word in tokens if len(word) < 21]
    # filter out stop words if requested
    if DROP_STOPWORDS:
        tokens = [w for w in tokens if not w in stoplist]         
    # perform word stemming if requested
    if STEMMING:
        ps = PorterStemmer()
        tokens = [ps.stem(word) for word in tokens]
    # recreate the document string from parsed words
    text = ''
    for token in tokens:
        text = text + ' ' + token
    return tokens, text 

##############################
### Gather Original Data 
##############################
newsgroups_all_train = fetch_20newsgroups(subset='train')
# pprint(list(newsgroups_all_train.target_names))
#print('\nnewsgroups_all_train.filenames.shape:', newsgroups_all_train.filenames.shape)
#print('\nnewsgroups_all_train.target.shape:', newsgroups_all_train.target.shape) 
#print('\nnewsgroups_all_train.target[:10]:', newsgroups_all_train.target[:10])

if SELECT_CATEGORY == 'COMPUTERS':
    categories = ['comp.graphics',
        'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware', 
        'comp.sys.mac.hardware',
        'comp.windows.x']

if SELECT_CATEGORY == 'RECREATION':
    categories = ['rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball', 
        'rec.sport.hockey']

if SELECT_CATEGORY == 'SCIENCE':
    categories = ['sci.crypt',
        'sci.electronics',
        'sci.med', 
        'sci.space']

if SELECT_CATEGORY == 'TALK':
    categories = ['talk.politics.guns',
        'talk.politics.mideast',
        'talk.politics.misc', 
        'talk.religion.misc']

if SELECT_CATEGORY == 'ALL':
    categories = newsgroups_all_train.target_names

#print('\nSelected newsgroups:')
#pprint(categories) 
# define set of training documents for the selected categories   
# remove headers, signature blocks, and quotation blocks from documents
newsgroups_train_original = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'),
                                      categories=categories)

#print('\nObject type of newsgroups_train_original.data:', 
#	type(newsgroups_train_original.data))    
#print('\nNumber of original training documents:',
#	len(newsgroups_train_original.data))	           
#print('\nFirst item from newsgroups_train.data_original\n', 
#	newsgroups_train_original.data[0])

# use generic name for target values
train_target = newsgroups_train_original.target

print(train_target.shape)
print(train_target[0])

