#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 21:22:38 2019

@author: paulhuynh
"""

###############################################################################
### packages required to run code.  Make sure to install all required packages.
###############################################################################
import re,string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import os

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import numpy as np

#Functionality to turn stemming on or off
STEMMING = False  # judgment call, parsed documents more readable if False
MAX_NGRAM_LENGTH = 2  # try 1 and 2 and see which yields better modeling results
VECTOR_LENGTH = 100  # set vector length for TF-IDF and Doc2Vec

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
        ps=PorterStemmer()
        tokens=[ps.stem(word) for word in tokens]
    return tokens


###############################################################################
### Processing text into lists
###############################################################################

# identify directory of POTUS text files (one file for each speech)
docdir = 'potus-files'
# os.chdir('/Users/paulhuynh/Desktop/run-potus-vectors/potus-files')

print('\nList of file names in the corpus:\n')
print(os.listdir(docdir))

#for loop to load documents
docs=[]
label=[]
year_list=[]
pattern='O_(.+?)_'
pattern2='_(.+?).txt'
#for file in os.listdir('.'):
for file in os.listdir(docdir):	
    if file.endswith('.txt'):
        with open(os.path.join(docdir,file), 'rb') as myfile:
            data=myfile.read().decode('utf-8')
            president=re.findall(pattern, file)[0] 
            year=re.findall(pattern2, file)[0][2:]
            docs.append(data)
            label.append(president)
            year_list.append(year)


corpus={'text':docs,'president':label, 'document_name': year_list}
#read in corpus into dataframe
data=pd.DataFrame(corpus)

#print to show the first and last 10 documents
print('\nGlimpse of beginning and end of corpus data frame:\n')
print(data.head(10))
print(data.tail(10))


#create empty list to store text documents labels
labels=[]

#for loop which appends the DSI title to the titles list
for i in range(0,len(data)):
    temp_text=data['document_name'].iloc[i]
    labels.append(temp_text)

#create empty list to store text documents
text_body=[]

#for loop which appends the text to the text_body list
for i in range(0,len(data)):
    temp_text=data['text'].iloc[i]
    text_body.append(temp_text)

    
#empty list to store processed documents
processed_text=[]
#for loop to process the text to the processed_text list
for i in text_body:
    text=clean_doc(i)
    processed_text.append(text)

#Note: the processed_text is the PROCESSED list of documents read directly from
#the csv.  Note the list of words is separated by commas.

#stitch back together individual words to reform body of text
final_processed_text=[]

for i in processed_text:
    temp_DSI=i[0]
    for k in range(1,len(i)):
        temp_DSI=temp_DSI+' '+i[k]
    final_processed_text.append(temp_DSI)
    
#the following is an example of what the processed text looks like.  
print('\nExample of what one parsed documnet looks like:\n')
print(final_processed_text[0])
    
#Note: We stitched the processed text together so the TFIDF vectorizer can work.
#Final section of code has 3 lists used.  2 of which are used for further processing.
#(1) text_body - unused, (2) processed_text (used in Word2Vec), 
#(3) final_processed_text (used in TFIDF), and (4) DSI titles (used in TFIDF Matrix)
 
###############################################################################
### Sklearn TFIDF 
###############################################################################
#note the ngram_range will allow you to include multiple-word tokens within the TFIDF matrix
#Call Tfidf Vectorizer
print('\nWorking on TF-IDF vectorization')
Tfidf=TfidfVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH), 
	max_features = VECTOR_LENGTH)

#fit the vectorizer using final processed documents.  The vectorizer requires the 
#stiched back together document.

TFIDF_matrix=Tfidf.fit_transform(final_processed_text)     

#creating datafram from TFIDF Matrix
matrix=pd.DataFrame(TFIDF_matrix.toarray(), 
	columns = Tfidf.get_feature_names(), 
	index = labels)

matrix.to_csv('tfidf-matrix.csv')
print('\nTF-IDF vectorization complete, matrix saved to tfidf-matrix.csv')

###############################################################################
### Explore TFIDF Values
###############################################################################
average_TFIDF={}
for i in matrix.columns:
    average_TFIDF[i]=np.mean(matrix[i])

average_TFIDF_DF = pd.DataFrame(average_TFIDF,
	index = [0]).transpose()

average_TFIDF_DF.columns=['TFIDF']

#calculate Q1 and Q3 range
Q1=np.percentile(average_TFIDF_DF, 25)
Q3=np.percentile(average_TFIDF_DF, 75)
IQR = Q3 - Q1
outlier=Q3+(1.5*IQR)

#words that exceed the Q3+IQR*1.5
outlier_list = average_TFIDF_DF[average_TFIDF_DF['TFIDF'] >= outlier]

#can export matrix to csv and explore further if necessary

###############################################################################
### Doc2Vec
###############################################################################
print("\nWorking on Doc2Vec vectorization")
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(final_processed_text)]
model = Doc2Vec(documents, vector_size = VECTOR_LENGTH, window = 2, 
	min_count = 1, workers = 4)

doc2vec_df=pd.DataFrame()
for i in range(0,len(processed_text)):
    vector=pd.DataFrame(model.infer_vector(processed_text[i])).transpose()
    doc2vec_df=pd.concat([doc2vec_df,vector], axis=0)

doc2vec_df=doc2vec_df.reset_index()

doc_titles={'title': labels}
t=pd.DataFrame(doc_titles)

doc2vec_df=pd.concat([doc2vec_df,t], axis=1)

doc2vec_df=doc2vec_df.drop('index', axis=1)
doc2vec_df=doc2vec_df.set_index('title')

doc2vec_df.to_csv('doc2vec-matrix.csv')
print('\nDoc2Vec vectorization complete, matrix saved to doc2vec-matrix.csv')

