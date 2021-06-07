# -*- coding: utf-8 -*-
"""
Created on Tue May  5 20:53:00 2020
Week 5: A.2 Second Research/Programming Assignment using manually generated classes/labels a.k.a y_train, y_pred

Created on Tue May  5 20:53:00 2020
This assignment concerns vectorization and document classification.
In this assignment, you can continue to work with your individual corpus or work with a corpus that you identify from the course or available public-domain sources. 
The Reduced Reuters Corpus may not be used for this assignment because extensive jump-start code is provided for that corpus. 
The corpus should have between two and ten identified classes of documents so that document classification can be performed as the final step of the study. 
The class of a document could be defined by the document source, with a known external variable, or with a variable that you, the analyst, define. 
It could be a subtopic within the general topic of interest used to define the corpus.
Consider three methods for assigning numerical vectors to documents. For each method, obtain a vector of numbers representing each document in the corpus. 
Represent these as row vectors, creating a documents-by-terms matrix for each vectorization method. We refer to the columns as "terms," but, 
depending on the method being employed, these could be individual words, n-grams, tokens, or (as is the case for Doc2Vec) index positions along a vector.

Approach 1: Analyst Judgment.

As we have reviewed in classroom discussions, initial work with document collections could begin with identifying 
important terms or equivalence classes (ECs) to be included in a corpus-wide Reference Term Vector (RTV). 
One way to do this is to employ analyst judgment guided by corpus statistics. 
To decide on whether or not we will keep a term in a small document collection, for example, we need to know that: 
    (1) It is important in at least one document, and 
    (2) It is prevalent in more than one document.
For larger document collections, we may specify percentages of documents in which we observe the terms or ECs. Analyst judgment is critical to this approach.
After the important terms have been identified, we can assign a number (perhaps a count or proportion) for each term in each document. 
That is, we can define a vector of numbers for each document.

Approach 2: TF-IDF.

Identify the top terms by corpus-wide statistics (TF-IDF, in particular). Regarding TF-IDF, we can compute the TF-IDF for 
each extracted term across the entire corpus. For our reference vector, we can choose a subset of terms with the highest TF-IDF values 
across the corpus. A high TF-IDF means that the term is both prevalent (across the corpus) and prominent (within at least one or more documents). 
Additionally, we have the TF-IDF value for each term within each document. Python Scikit Learn provides TF-IDF vectorization:

https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html (Links to an external site.)

Approach 3: Neural Network Embeddings (Doc2Vec).

With this approach, we utilize machine learning methods to convert documents to vectors of numbers. Such methods draw on self-supervised 
machine learning (autoencoding a la Word2Vec). Instead of Word2Vec, however, we use Doc2Vec, representing each document with a set of numbers. 
The numbers come from neural network weights or embeddings. The numbers are not directly associated with terms, 
so the meaning of the numbers is undefined. Python Gensim provides Doc2Vec vectorizations:

https://radimrehurek.com/gensim/models/doc2vec.html (Links to an external site.)

Management Problem. Part of your job in this assignment is to define a meaningful management problem. 
The corpus you use should relate in some way to a business, organizational, or societal problem. 
Regardless of the neural network methods being employed, research results should 
provide guidance in addressing the management problem. 
Research Methods/Programming Components

This research/programming assignment involves ten activities as follows:

(1) Define a management goal for your research. What do you hope to learn from this natural language study? 
    What management questions will be addressed? Consider a goal related to document classification.
(2) Identify the individual corpus you will be using in the assignment. The corpus should 
    be available as a JSON lines file. Previously, we had suggested that the JSON lines file be set up 
    with at least four key-value pairs defined as "ID," "URL," "TITLE,", and "BODY," 
    where "BODY" represents a plain text document. To facilitate subsequent analyses, 
    it may be convenient to use a short character string (say, eight characters or less) 
    to identify each document. This short character string could be the value 
    associated with the ID key or with an additional key that you define. 
(3) Preprocess the text documents, ensuring that unnecessary tags, punctuation, and images are excluded.  
(4) Create document vectors using Approach 1 above.
(5) Create document vectors using Approach 2 above.
(6) Create document vectors using Approach 3 above.
(7) Compare results across the three approaches. In comparing Approach 1 with Approach 2, 
    for example, find the two or three terms (nouns/noun phrases) from your documents 
    that you thought to be important/prevalent from Approach 1 and see if they did indeed 
    have the highest TF-IDF as shown in the results from Approach 2. 
    Similarly, find two or three terms that you thought would have a lower 
    importance/prevalence, and see if that bears out. Judge the degree of agreement across the approaches.
(8) Review results in light of the management goal for this research. Do you have concerns about 
    the corpus? Are there ways that the corpus should be extended or contracted in order to address management questions?
(9) Prepare numerical matrices for further analysis. For each of the three vectorization approaches, 
    construct a matrix of real numbers with rows representing documents 
    (i.e. each row is a numerical vector representing a document). 
    We could refer to each matrix as a documents-by-terms matrix 
    (although the elements of vectors from Approach 3 are not directly associated with terms). 
    To facilitate future research with these matrices, rows should be associated with short 
    character strings representing the documents in the corpus. Also, for Approaches 1 and 2, 
    it will be convenient to have columns identified by character strings for the terms. 
    If the terms are n-grams (groups of n words in sequence), it may be a good idea to 
    replace blank characters with underlines, so that the columns may be 
    interpreted as variable names in modeling programs.
(10) Working with the two to ten classes of documents for this exercise, 
    use a random forest classifier to fit three text classification models, 
    one for each of the vectorization methods. Determine the vectorization 
    method that does the best job of classification based on an index of classification accuracy. 
    If the number of documents is large or if classes have not been identified in advance, 
    select a subset of the documents under study (perhaps 100 or 200), 
    identify each document with a class or category. An example of this type 
    of analysis is shown in example jump-start code under

@author: Ezana.Beyenne
"""

import multiprocessing
import re,string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import pandas as pd


import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


#Functionality to turn stemming on or off
STEMMING = True  # judgment call, parsed documents more readable if False
MAX_NGRAM_LENGTH = 1  # try 1 and 2 and see which yields better modeling results
VECTOR_LENGTH = 100  # set vector length for TF-IDF and Doc2Vec
DISPLAYMAX = 10 # Dispaly count for head() or tail() sorted values
DROP_STOPWORDS = False
SET_RANDOM = 9999

##############################################################################
### Utility Functions 
##############################################################################
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
       lem = WordNetLemmatizer()
       tokens = [lem.lemmatize(word) for word in tokens]
    # recreate the document string from parsed words
    text = ''
    for token in tokens:
        text = text + ' ' + token
    return tokens, text 

def create_label(text):
    #print(text)
    text = text.replace('.html','')
    #print(text)
    text = text.replace('.htm','')
    if 'wired-' in text:
        text = text.replace('wired-','w-')
    elif 'nhtsa-' in text:
        text = text.replace('nhtsa-', 'n-')
    elif 'curbed-' in text:
        text = text.replace('curbed-','c-')
    elif 'theverge-' in text:
        text = text.replace('theverge-','v-')
    regex = re.compile('[^a-zA-Z]')
    regex.sub('', text)
    return text[0:12]

#######################################################################
# Analyst Judgement to manually identify the important terms (classes) 
#    and their equivalent classes 
#######################################################################
def get_y_class(text):     
    safety = len(re.findall('safety',text))
    technology = len(re.findall('technology',text))
    crash = len(re.findall('crash',text))
    people = len(re.findall('people',text))
    driver = len(re.findall('driver',text))
    human= len(re.findall('human',text))
    lidar= len(re.findall('lidar',text))
    autonomous= len(re.findall('autonomous',text))
    research = len(re.findall('research ',text))
    selfdriving  = len(re.findall('selfdriving ',text))
    policy  = len(re.findall('policy ',text))
    standard  = len(re.findall('standard ',text))
    rule  = len(re.findall('rule ',text))
    government = len(re.findall('government ',text))
    testing = len(re.findall('testing ',text))
    electric  = len(re.findall('electric ',text))
    engineer  = len(re.findall('engineer ',text))
    system  = len(re.findall('system ',text))    
    
    # Safety     = safety, crash, people, driver, human and policy
    # Technology = technology, libar, autonomous, research, selfdriving
    safetyClassCount = safety + crash + people + driver + human + policy + standard + rule + government + testing
    technologyClassCount = technology + lidar + autonomous + research + selfdriving + electric + engineer +  system
    
    # Safety class == 0, Technology class = 1
    if safetyClassCount < technologyClassCount:
       return 1
   
    return 0


labels=[]
text_body=[]
text_titles = []
regex = re.compile('[^a-zA-Z]')
with open('autonomous_vehicles_safety_corpus.jl') as json_file:
     data = json.load(json_file)
     for p in data:
         text_body.append(p['BODY'])
         text_titles.append(p['TITLE'][0:8])
         labels.append(create_label(p['FILENAME'][0:8]))
 

#############################################################################
# generate split of train and test data and
# random sample by that split data by k number
#############################################################################
def train_test_split_with_random_sampling(text_body, k):
    # list of token lists for gensim Doc2Vec
    train_tokens = []
    # list of document strings for sklearn TF-IDF
    train_text = [] 
    # list of y variables from class labels analyst judgement
    train_target = [] 
    train = random.sample(text_body, k)
    
    for doc in train:
        text_string = doc
        # parse the entire document string
        text_string = parse_doc(text_string)
        # parse words one at a time in document string
        tokens, text_string = parse_words(text_string)
        train_tokens.append(tokens)
        train_text.append(text_string)
        train_target.append(get_y_class(text_string))
                
    return train_tokens, train_text, train_target
        
    
##############################################################################
### Prepare Training and test Data random sampling 80% training sample
##############################################################################
train_k =int(len(text_body) * 0.8)
test_k = int(len(text_body) - train_k)
train_tokens, train_text, train_target = train_test_split_with_random_sampling(text_body, train_k) 
test_tokens, test_text, test_target = train_test_split_with_random_sampling(text_body, test_k) 
         

##############################################################################
#  Number of cpu cores
##############################################################################
cores = multiprocessing.cpu_count()
print("\nNumber of processor cores:", cores)


##############################
### Count Vectorization
##############################

print('\nCount Vectorization. . .')
count_vectorizer = CountVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH), 
    max_features = VECTOR_LENGTH)
count_vectors = count_vectorizer.fit_transform(train_text)

#print('\nTraining count_vectors_training.shape:', count_vectors.shape)

# Apply the same vectorizer to the test data
# Notice how we use count_vectorizer.transform, 
#                      NOT count_vectorizer.fit_transform
count_vectors_test = count_vectorizer.transform(test_text)
#print('\nTest count_vectors_test.shape:', count_vectors_test.shape)
count_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
	random_state = SET_RANDOM)
count_clf.fit(count_vectors, train_target)

# evaluate on test set
count_pred = count_clf.predict(count_vectors_test)  

cv_RF_F1 = round(metrics.f1_score(test_target, count_pred, average='macro'), 3)
#print('\nCount/Random forest F1 classification performance in test set:', cv_RF_F1)
   

##############################
### TF-IDF Vectorization
##############################
print('\nTFIDF vectorization. . .')

tfidf_vectorizer = TfidfVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH), 
    max_features = VECTOR_LENGTH)
tfidf_vectors = tfidf_vectorizer.fit_transform(train_text)

#print('\nTraining tfidf_vectors_training.shape:', tfidf_vectors.shape)

# Apply the same vectorizer to the test data
# Notice how we use tfidf_vectorizer.transform, NOT 
# tfidf_vectorizer.fit_transform
tfidf_vectors_test = tfidf_vectorizer.transform(test_text)
#print('\nTest tfidf_vectors_test.shape:', tfidf_vectors_test.shape)
tfidf_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
	                               random_state = SET_RANDOM)

tfidf_clf.fit(tfidf_vectors, train_target)

# evaluate on test set
tfidf_pred = tfidf_clf.predict(tfidf_vectors_test) 
tfidf_RF_F1 = round(metrics.f1_score(test_target, tfidf_pred, average='macro'), 3) 
#print('\nTF-IDF/Random forest F1 classification performance in test set:', tfidf_RF_F1)


###########################################
### Doc2Vec Vectorization (50 dimensions)
###########################################
print('\nDoc2Vec Vectorization (50 dimensions). . .')

train_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_tokens)]
#print('train_corpus[:2]:', train_corpus[:1])

#print("\nWorking on Doc2Vec vectorization, dimension 50")
model_50 = Doc2Vec(vector_size = 50, window = 4, 
	min_count = 2, workers = cores, epochs = 40)
model_50.build_vocab(train_corpus)

# build vectorization model on training set
model_50.train(train_corpus, total_examples = model_50.corpus_count, 
	epochs = model_50.epochs)  

# vectorization for the training set
# initialize numpy array
doc2vec_50_vectors = np.zeros((len(train_corpus), 50)) 

for i in range(0, len(train_tokens)):
    doc2vec_50_vectors[i,] = model_50.infer_vector(train_tokens[i]).transpose()   
    
#print('\nTraining doc2vec_50_vectors.shape:', doc2vec_50_vectors.shape)
#print('doc2vec_50_vectors[:2]:', doc2vec_50_vectors[:2])

# vectorization for the test set
# initialize numpy array
doc2vec_50_vectors_test = np.zeros((len(test_tokens), 50)) 

for i in range(0, len(test_tokens)):
    doc2vec_50_vectors_test[i,] = model_50.infer_vector(test_tokens[i]).transpose()

#print('\nTest doc2vec_50_vectors_test.shape:', doc2vec_50_vectors_test.shape)

doc2vec_50_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
                                        random_state = SET_RANDOM)

# fit model on training set
doc2vec_50_clf.fit(doc2vec_50_vectors, train_target) 

# evaluate on test set
doc2vec_50_pred = doc2vec_50_clf.predict(doc2vec_50_vectors_test) 

# print the F1 score
doc2vec_50_RF_F1 = round(metrics.f1_score(test_target, doc2vec_50_pred, average='macro'), 3)
#print('\nDoc2Vec_50/Random forest F1 classification performance in test set:', doc2vec_50_RF_F1) 

###########################################
### Doc2Vec Vectorization (100 dimensions)
###########################################
print('\nDoc2Vec Vectorization (100 dimensions). . .')

model_100 = Doc2Vec(train_corpus, vector_size = 100, window = 4, 
	min_count = 2, workers = cores, epochs = 40)

model_100.train(train_corpus, total_examples = model_100.corpus_count, 
	epochs = model_100.epochs)  # build vectorization model on training set

# vectorization for the training set
# initialize numpy array
doc2vec_100_vectors = np.zeros((len(train_tokens), 100)) 

for i in range(0, len(train_tokens)):
    doc2vec_100_vectors[i,] = model_100.infer_vector(train_tokens[i]).transpose()
    
#print('\nTraining doc2vec_100_vectors.shape:', doc2vec_100_vectors.shape)


# vectorization for the test set
# initialize numpy array
doc2vec_100_vectors_test = np.zeros((len(test_tokens), 100))

for i in range(0, len(test_tokens)):
    doc2vec_100_vectors_test[i,] = model_100.infer_vector(test_tokens[i]).transpose()
    
#print('\nTest doc2vec_100_vectors_test.shape:', doc2vec_100_vectors_test.shape)

doc2vec_100_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
                                         random_state = SET_RANDOM)

# fit model on training set
doc2vec_100_clf.fit(doc2vec_100_vectors, train_target) 

# evaluate on test set
doc2vec_100_pred = doc2vec_100_clf.predict(doc2vec_100_vectors_test)  

# print the F1 score

doc2vec_100_RF_F1 =  round(metrics.f1_score(test_target, doc2vec_100_pred, average='macro'), 3)
#print('\nDoc2Vec_100/Random forest F1 classification performance in test set:', doc2vec_100_RF_F1) 

###########################################
### Doc2Vec Vectorization (200 dimensions)
###########################################
print('\nDoc2Vec Vectorization (200 dimensions). . .')

model_200 = Doc2Vec(train_corpus, vector_size = 200, window = 4, 
	min_count = 2, workers = cores, epochs = 40)

# build vectorization model on training set
model_200.train(train_corpus, total_examples = model_200.corpus_count, 
	epochs = model_200.epochs) 

# vectorization for the training set
# initialize numpy array
doc2vec_200_vectors = np.zeros((len(train_tokens), 200)) 

for i in range(0, len(train_tokens)):
    doc2vec_200_vectors[i,] = model_200.infer_vector(train_tokens[i]).transpose()
    
#print('\nTraining doc2vec_200_vectors.shape:', doc2vec_200_vectors.shape)
# print('doc2vec_200_vectors[:2]:', doc2vec_200_vectors[:2])

# vectorization for the test set
# initialize numpy array
doc2vec_200_vectors_test = np.zeros((len(test_tokens), 200)) 

for i in range(0, len(test_tokens)):
    doc2vec_200_vectors_test[i,] = model_200.infer_vector(test_tokens[i]).transpose()
    
#print('\nTest doc2vec_200_vectors_test.shape:', doc2vec_200_vectors_test.shape)

doc2vec_200_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
                                         random_state = SET_RANDOM)

# fit model on training set
doc2vec_200_clf.fit(doc2vec_200_vectors, train_target) 

# evaluate on test set
doc2vec_200_pred = doc2vec_200_clf.predict(doc2vec_200_vectors_test) 

# print the F1 score
doc2vec_200_RF_F1 = round(metrics.f1_score(test_target, doc2vec_200_pred, average='macro'), 3)

#print('\nDoc2Vec_200/Random forest F1 classification performance in test set:', doc2vec_200_RF_F1) 


print('\n------------------------------------------------------------------------')
print('\nTF-IDF/Random forest F1 classification performance in test set:', cv_RF_F1 )
print('\nCount/Random forest F1 classification performance in test set:', tfidf_RF_F1)
print('\nDoc2Vec_50/Random forest F1 classification performance in test set:',  doc2vec_50_RF_F1) 
print('\nDoc2Vec_100/Random forest F1 classification performance in test set:', doc2vec_100_RF_F1)   
print('\nDoc2Vec_200/Random forest F1 classification performance in test set:', doc2vec_200_RF_F1) 
print('\n------------------------------------------------------------------------')

df = pd.DataFrame(data = [[cv_RF_F1],
                   [tfidf_RF_F1],
                   [doc2vec_50_RF_F1],
                   [doc2vec_100_RF_F1],
                   [doc2vec_200_RF_F1]],
                    columns=['F1 classification performance in test set with manual labeling'],
                     index=['TF-IDF/Random forest classification',
                            'CountVec/Random forest classification',
                            'Doc2Vec_50/Random forest classification',
                            'Doc2Vec_100/Random forest classification',
                            'Doc2Vec_200/Random forest classification'])
df.index.name ='Algorithm'
df = df.sort_values('F1 classificaton performance in test set with manual labeling', ascending=False)
print(df)