# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:18:35 2020
Week 5: A.2 Second Research/Programming Assignment using k-means generated labels

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
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, KFold

import pandas as pd
import os

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler

import numpy as np
from nltk.stem import WordNetLemmatizer
#Functionality to turn stemming on or off
STEMMING = True  # judgment call, parsed documents more readable if False
MAX_NGRAM_LENGTH = 1  # try 1 and 2 and see which yields better modeling results
VECTOR_LENGTH = 100  # set vector length for TF-IDF and Doc2Vec
DISPLAYMAX = 10 # Dispaly count for head() or tail() sorted values
DROP_STOPWORDS = False
SET_RANDOM = 9999


##############################################################################
#  Number of cpu cores
##############################################################################
cores = multiprocessing.cpu_count()
print("\nNumber of processor cores:", cores)

#############################################################################
#  Create Labels 
############################################################################
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
    return text[0:8]

###############################################################################
### Function to process documents
###############################################################################
def clean_doc(doc): 
    # split document into individual words
    tokens=doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # # filter out short tokens
    tokens = [word for word in tokens if len(word) > 4]
    # #lowercase all words
    tokens = [word.lower() for word in tokens]
    # # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]         
    # # word stemming Commented
    if STEMMING:   
        lem = WordNetLemmatizer()
        tokens = [lem.lemmatize(token) for token in tokens]
    return tokens


labels=[]
text_body=[]
text_titles = []
regex = re.compile('[^a-zA-Z]')
with open('autonomous_vehicles_safety_corpus.jl') as json_file:
     data = json.load(json_file)
     for p in data:
         text_body.append(p['BODY'])
         text_titles.append(p['TITLE'][0:8])
         labels.append(create_label(p['FILENAME']))
         
 
################################
# K Means to get the labels
############################
#empty list to store processed documents
processed_text=[]
#for loop to process the text to the processed_text list
for i in text_body:
    text=clean_doc(i)
    processed_text.append(text)
    
#stitch back together individual words to reform body of text
final_processed_text=[]

for i in processed_text:
    temp_DSI=i[0]
    for k in range(1,len(i)):
        temp_DSI=temp_DSI+' '+i[k]
    final_processed_text.append(temp_DSI)
    

##############################
### TF-IDF Vectorization
##############################
# run tfidf (prevalent - require 25% of docs)

print('\nTF-IDF Vectorization')

tfidf = TfidfVectorizer(ngram_range=(1,1), min_df=0.25)
tfidf_matrix = tfidf.fit_transform(final_processed_text)

print('\n\t\tTF-IDF Vectorization. . .')
#print('\nTraining tfidf_matrix.shape:', tfidf_matrix.shape)

print('\nTF-IDF Vectorization K Means vectorization. . .')
k=3
km = KMeans(n_clusters=k, random_state=89)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
#print(clusters)

y = clusters
X = tfidf_matrix

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=23)
print( x_train.shape)


# Apply the same vectorizer to the test data
# Notice how we use tfidf_vectorizer.transform, NOT tfidf_vectorizer.fit_transform
tfidf_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
	random_state = SET_RANDOM)

tfidf_clf.fit(x_train, y_train)
tfidf_pred = tfidf_clf.predict(x_val)  # evaluate on test set

tfidf_RF_F1 = round(metrics.f1_score(y_val, tfidf_pred, average='macro'), 3)

#####################################################################################################
### Count Vectorization
#####################################################################################################
print('\n\t\tCount Vectorization')

count_vectorizer = CountVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH), max_features = VECTOR_LENGTH)
count_vectors_matrix = count_vectorizer.fit_transform(final_processed_text)

print('\nCount Vectorization K Means vectorization. . .')

kmCV = KMeans(n_clusters=k, random_state=89)
kmCV.fit(count_vectors_matrix)
clustersCv = kmCV.labels_.tolist()

y1 = clustersCv
X1 = count_vectors_matrix

x_trainC, x_valC, y_trainC, y_valC = train_test_split(X1, y1, test_size=0.2, random_state=23)
#print( x_train.shape)

count_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = SET_RANDOM)
count_clf.fit(x_trainC, y_trainC)
count_pred = count_clf.predict(x_valC)  # evaluate on test set

cv_RF_F1 =  round(metrics.f1_score(y_valC, count_pred, average='macro'), 3)


##################################################################################################
# train_corpus using TaggedDocument
##################################################################################################
train_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(processed_text)]
#print('train_corpus[:2]:', train_corpus[:1])

#################################################################################################
### Doc2Vec Vectorization (50 dimensions)
#################################################################################################

print("\n\t\tWorking on Doc2Vec vectorization, dimension 50")

model_50 = Doc2Vec(vector_size = 50, window = 4, min_count = 2, workers = cores, epochs = 40)
model_50.build_vocab(train_corpus)
model_50.train(train_corpus, total_examples = model_50.corpus_count, 
	epochs = model_50.epochs)  # build vectorization model on training set

# vectorization for the training set
doc2vec_50_vectors = np.zeros((len(train_corpus), 50)) # initialize numpy array

for i in range(0, len(processed_text)):
    doc2vec_50_vectors[i,] = model_50.infer_vector(processed_text[i]).transpose()
      
#print('\nTraining doc2vec_50_vectors.shape:', doc2vec_50_vectors.shape)

print('\nDoc2Vec 50 Vectorization K Means vectorization. . .')
kmDV = KMeans(n_clusters=k, random_state=89)
kmDV.fit(doc2vec_50_vectors)
clustersDV = kmDV.labels_.tolist()

y1 = clustersCv
X1 = doc2vec_50_vectors

x_traind, x_vald, y_traind, y_vald = train_test_split(X1, y1, test_size=0.2, random_state=SET_RANDOM)
print( x_train.shape)

doc2vec_50_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = SET_RANDOM)
# fit model on training set
doc2vec_50_clf.fit(x_traind, y_traind) 

# evaluate on test set
doc2vec_50_pred = doc2vec_50_clf.predict(x_vald)  
doc2vec_50_RF_F1 = round(metrics.f1_score(y_vald, doc2vec_50_pred, average='macro'), 3)


#################################################################################################
### Doc2Vec Vectorization (100 dimensions)
#################################################################################################

print("\n\t\tWorking on Doc2Vec vectorization, dimension 100")

model_100 = Doc2Vec(vector_size = 100, window = 4, 
	min_count = 2, workers = cores, epochs = 40)
model_100.build_vocab(train_corpus)

# build vectorization model on training set
model_100.train(train_corpus, total_examples = model_100.corpus_count, epochs = model_100.epochs)  

# vectorization for the training set
doc2vec_100_vectors = np.zeros((len(train_corpus), 100)) # initialize numpy array

for i in range(0, len(processed_text)):
    doc2vec_100_vectors[i,] = model_100.infer_vector(processed_text[i]).transpose()
      

print('\nDoc2Vec 100 Vectorization K Means vectorization. . .')
kmDV = KMeans(n_clusters=k, random_state=89)
kmDV.fit(doc2vec_100_vectors)
clustersDV = kmDV.labels_.tolist()

y1 = clustersCv
X1 = doc2vec_100_vectors

x_traind, x_vald, y_traind, y_vald = train_test_split(X1, y1, test_size=0.2, random_state=SET_RANDOM)
print( x_train.shape)

doc2vec_100_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = SET_RANDOM)
# fit model on training set
doc2vec_100_clf.fit(x_traind, y_traind) 

# evaluate on test set
doc2vec_100_pred = doc2vec_100_clf.predict(x_vald)  
doc2vec_100_RF_F1 = round(metrics.f1_score(y_vald, doc2vec_100_pred, average='macro'), 3)

#################################################################################################
### Doc2Vec Vectorization (200 dimensions)
#################################################################################################

print("\n\t\tWorking on Doc2Vec vectorization, dimension 200")

model_200 = Doc2Vec(vector_size = 200, window = 4, min_count = 2, workers = cores, epochs = 40)
model_200.build_vocab(train_corpus)

# build vectorization model on training set
model_200.train(train_corpus, total_examples = model_200.corpus_count, epochs = model_200.epochs)  

# vectorization for the training set
doc2vec_200_vectors = np.zeros((len(train_corpus), 200)) # initialize numpy array

for i in range(0, len(processed_text)):
    doc2vec_200_vectors[i,] = model_200.infer_vector(processed_text[i]).transpose()
      

print('\nDoc2Vec 200 Vectorization K Means vectorization. . .')
kmDV = KMeans(n_clusters=k, random_state=89)
kmDV.fit(doc2vec_200_vectors)
clustersDV = kmDV.labels_.tolist()

y1 = clustersCv
X1 = doc2vec_200_vectors

x_traind, x_vald, y_traind, y_vald = train_test_split(X1, y1, test_size=0.2, random_state=SET_RANDOM)
print( x_train.shape)

doc2vec_200_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = SET_RANDOM)
# fit model on training set
doc2vec_200_clf.fit(x_traind, y_traind) 

# evaluate on test set
doc2vec_200_pred = doc2vec_200_clf.predict(x_vald)  
doc2vec_200_RF_F1 = round(metrics.f1_score(y_vald, doc2vec_200_pred, average='macro'), 3)

#######################################################################################################
#  Output the results to a DataFrame
#######################################################################################################
df = pd.DataFrame(data = [[tfidf_RF_F1],
                   [cv_RF_F1],
                   [doc2vec_50_RF_F1],
                   [doc2vec_100_RF_F1],
                   [doc2vec_200_RF_F1]],
                    columns=['F1 classification performance in test set with k-means labeling'],
                     index=['TF-IDF/Random forest classification',
                            'CountVec/Random forest classification',
                            'Doc2Vec_50/Random forest classification',
                            'Doc2Vec_100/Random forest classification',
                            'Doc2Vec_200/Random forest classification'])
df.index.name ='Algorithm'
df = df.sort_values('F1 classification performance in test set with k-means labeling', ascending=False)
print(df)
