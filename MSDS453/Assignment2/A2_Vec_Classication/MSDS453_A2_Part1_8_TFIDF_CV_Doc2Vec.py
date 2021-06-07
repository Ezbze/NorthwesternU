"""
Week 5: A.2 Second Research/Programming Assignment

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

With this approach, we utilize machine learning methods to convert documents to vectors of numbers. Such methods draw on self-supervised machine learning (autoencoding a la Word2Vec). Instead of Word2Vec, however, we use Doc2Vec, representing each document with a set of numbers. The numbers come from neural network weights or embeddings. The numbers are not directly associated with terms, so the meaning of the numbers is undefined. Python Gensim provides Doc2Vec vectorizations:

https://radimrehurek.com/gensim/models/doc2vec.html (Links to an external site.)

Management Problem. Part of your job in this assignment is to define a meaningful management problem. The corpus you use should relate in some way to a business, organizational, or societal problem. Regardless of the neural network methods being employed, research results should provide guidance in addressing the management problem. 
Research Methods/Programming Components

This research/programming assignment involves ten activities as follows:

(1) Define a management goal for your research. What do you hope to learn from this natural language study? What management questions will be addressed? Consider a goal related to document classification.
(2) Identify the individual corpus you will be using in the assignment. The corpus should be available as a JSON lines file. Previously, we had suggested that the JSON lines file be set up with at least four key-value pairs defined as "ID," "URL," "TITLE,", and "BODY," where "BODY" represents a plain text document. To facilitate subsequent analyses, it may be convenient to use a short character string (say, eight characters or less) to identify each document. This short character string could be the value associated with the ID key or with an additional key that you define. 
(3) Preprocess the text documents, ensuring that unnecessary tags, punctuation, and images are excluded.  
(4) Create document vectors using Approach 1 above.
(5) Create document vectors using Approach 2 above.
(6) Create document vectors using Approach 3 above.
(7) Compare results across the three approaches. In comparing Approach 1 with Approach 2, for example, find the two or three terms (nouns/noun phrases) from your documents that you thought to be important/prevalent from Approach 1 and see if they did indeed have the highest TF-IDF as shown in the results from Approach 2. Similarly, find two or three terms that you thought would have a lower importance/prevalence, and see if that bears out. Judge the degree of agreement across the approaches.
(8) Review results in light of the management goal for this research. Do you have concerns about the corpus? Are there ways that the corpus should be extended or contracted in order to address management questions?
@author: Ezana.Beyenne
"""

import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


import pandas as pd
import multiprocessing
import glob
from nltk import *
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from collections import Counter 
import collections
import numpy as np
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#Functionality to turn stemming on or off
STEMMING = True  # judgment call, parsed documents more readable if False
MAX_NGRAM_LENGTH = 1  # try 1 and 2 and see which yields better modeling results
VECTOR_LENGTH = 100  # set vector length for TF-IDF and Doc2Vec
DISPLAYMAX = 6 # Dispaly count for head() or tail() sorted values
DROP_STOPWORDS = False
SET_RANDOM = 9999

# Display the dataframes side by side
from IPython.display import display, HTML

CSS = """
.output {
    flex-direction: row;
}
"""

HTML('<style>{}</style>'.format(CSS))


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

#(3) Preprocess the text documents, ensuring that unnecessary tags, punctuation, and images are excluded.  
def clean_doc(doc):
    """Return processed tokens for a given document."""
    # Split into "words"
    tokens = doc.split()
    # Remove punctuation
    re_punc = re.compile(f"[{re.escape(string.punctuation)}]")
    tokens = [re_punc.sub('', word) for word in tokens]
    # Remove non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]
    # Remove short tokens
    tokens = [word for word in tokens if len(word) > 4]
    # Make tokens lowercase
    tokens = [word.lower() for word in tokens]
    # Remove stop words
    stop_words = stopwords.words('english')
    stop_words.append("would")
    tokens = [word for word in tokens if word not in stop_words]
    # lemmatization for plurals  
    if STEMMING:   
        lem = WordNetLemmatizer()
        tokens = [lem.lemmatize(token) for token in tokens]
    return tokens

# Extract the data
labels=[]
text_body=[]
text_titles = []
with open('autonomous_vehicles_safety_corpus.jl') as json_file:
     data = json.load(json_file)
     for p in data:
         text_body.append(p['BODY'])
         text_titles.append(p['TITLE'][0:8])
         labels.append(create_label(p['FILENAME']))

# clean words and create list of tokens
processed_text = []
for document in text_body:
    processed_text.append(clean_doc(document))
    
# Stitch the clean data 
final_processed_text = []  
for i in processed_text:
    temp_DSI=i[0]
    for k in range(1,len(i)):
        temp_DSI=temp_DSI+' '+i[k]
    final_processed_text.append(temp_DSI)
    

##############################
### Count Vectorization Frequency
##############################
count_vectorizer = CountVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH), 
                                   max_features = VECTOR_LENGTH)
count_vectors = count_vectorizer.fit_transform(final_processed_text)
#print('\ncount vectorization. . .')
#print('\nTraining count_vectors_training.shape:', count_vectors.shape)

matrixAnalystJudgment = pd.DataFrame(count_vectors.toarray(), columns=count_vectorizer.get_feature_names(), index=labels)

#(4) Create document vectors using Approach 1 above. output Document Frequency
###############################################################################
### Explore CountVectorizer Values
###############################################################################
#print('\nWorking on Count vectorization')
average_CountVectorizer={}
for i in matrixAnalystJudgment.columns:
    average_CountVectorizer[i]=np.mean(matrixAnalystJudgment[i])

average_CountVectorizer_DF = pd.DataFrame(average_CountVectorizer, index = [0]).transpose()

average_CountVectorizer_DF.columns=['A1-Freq']

#calculate Q1 and Q3 range
Q=np.percentile(average_CountVectorizer_DF, 5)
Q1=np.percentile(average_CountVectorizer_DF, 25)
Q3=np.percentile(average_CountVectorizer_DF, 75)
IQR = Q3 - Q1
outlier=Q3+(1.5*IQR)

#words that exceed the Q3+IQR*1.5
outlier_list = average_CountVectorizer_DF[average_CountVectorizer_DF['A1-Freq'] >= outlier]
#print(outlier_list.sort_values('CountVectorizer'))

sorted_CountVectorizer = average_CountVectorizer_DF.sort_values('A1-Freq', ascending = False)

#print('\nLargest Count vectors')
sorted_CountVectorizer.index.name = 'Terms'
#print(sorted_CountVectorizer.head(DISPLAYMAX))

# A2. using TF-IDF.
# (5) Create document vectors using Approach 2 above.

###############################################################################
### Sklearn TFIDF 
###############################################################################
#note the ngram_range will allow you to include multiple-word tokens within the TFIDF matrix
#Call Tfidf Vectorizer
#print('\nWorking on TF-IDF vectorization')

Tfidf=TfidfVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH), max_features = VECTOR_LENGTH)

#fit the vectorizer using final processed documents.  The vectorizer requires the 
#stiched back together document.

TFIDF_matrix=Tfidf.fit_transform(final_processed_text)     

#creating datafram from TFIDF Matrix
matrix=pd.DataFrame(TFIDF_matrix.toarray(), columns = Tfidf.get_feature_names(), index = labels)


###############################################################################
### Explore TFIDF Values
###############################################################################
average_TFIDF={}
for i in matrix.columns:
    average_TFIDF[i]=np.mean(matrix[i])

average_TFIDF_DF = pd.DataFrame(average_TFIDF, index = [0]).transpose()

average_TFIDF_DF.columns=['TFIDF-Freq']

#calculate Q1 and Q3 range

Q1=np.percentile(average_TFIDF_DF, 25)
Q3=np.percentile(average_TFIDF_DF, 75)
IQR = Q3 - Q1
outlier=Q3+(1.5*IQR)

# A2 - showing document frequency
#words that exceed the Q3+IQR*1.5
outlier_list_TFIDF = average_TFIDF_DF[average_TFIDF_DF['TFIDF-Freq'] >= outlier]
#print(outlier_list_TFIDF.sort_values('TFIDF'))
sortedTf_IDF = average_TFIDF_DF.sort_values('TFIDF-Freq', ascending = False)
sortedTf_IDF.index.name = 'Terms'
#print('\nLargest TF-IDF vectors')
#print(sortedTf_IDF.head(DISPLAYMAX))

# (6) Create document vectors using Approach 3 above.
###########################################
### Doc2Vec Vectorization
###########################################

#print('Begin Doc2Vec Work')
cores = multiprocessing.cpu_count()
#print("Number of processor cores:", cores)

train_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(processed_text)]
#print('train_corpus[:2]:', train_corpus[:1])

#print("\nWorking on Doc2Vec vectorization.")
model_d2v = Doc2Vec(vector_size = VECTOR_LENGTH, window = 4, min_count = 2, workers = cores, epochs = 40)

model_d2v.build_vocab(train_corpus)

# build vectorization model on training set
model_d2v.train(train_corpus, total_examples = model_d2v.corpus_count, epochs = model_d2v.epochs)  

doc2VecToList = []
for word, vocab_obj in model_d2v.wv.vocab.items():
    doc2VecToList.append([word,vocab_obj.count])

df_doc2_vec = pd.DataFrame(doc2VecToList, columns=['Terms', 'A3-Doc2Vec-Count'])
df_doc2_vec.set_index('Terms', inplace=True)
df_doc2_vec[df_doc2_vec['A3-Doc2Vec-Count'] > 1500]

df_doc2Vec =df_doc2_vec.sort_values('A3-Doc2Vec-Count', ascending=False)


#(7) Compare results across the three approaches. 
#      In comparing Approach 1 with Approach 2, for example, 
#       find the two or three terms (nouns/noun phrases) 
#       from your documents that you thought to be important/prevalent 
#       from Approach 1 and see if they did indeed have the highest TF-IDF 
#       as shown in the results from Approach 2. 
#       Similarly, find two or three terms that you thought would have a lower importance/prevalence, 
#       and see if that bears out. 
#     Judge the degree of agreement across the approaches.
print('\n\nDataFrame of the most important 6 terms.\n')
print('\nA1. Analyst Judgement derived important terms using CountVectorizer.\n')
print(sorted_CountVectorizer.head(DISPLAYMAX))
print('\nA2.TF-IDF derived important terms.\n')
print(sortedTf_IDF.head(DISPLAYMAX))
print('\nA2.Doc2Vec derived important terms.\n')
print(df_doc2Vec.head(DISPLAYMAX))


#(7) Similarly, find two or three terms that  
#    you thought would have a lower importance/prevalence,and see if that bears out. 
#    Judge the degree of agreement across the approaches.
print('\n\nDataFrame of the lower importance 6 terms.\n')
print('\nA1. Analyst Judgement derived least terms using CountVectorizer.\n')
print(sorted_CountVectorizer.tail(DISPLAYMAX))
print('\nA2.TF-IDF derived least important terms.\n')
print(sortedTf_IDF.tail(DISPLAYMAX))
print('\nA2.Doc2Vec derived least important terms.\n')
print(df_doc2Vec.tail(DISPLAYMAX))



## Manual Labels and Equivalent Terms
safetyCount = 0
techCount = 0
counter = 0
equal = 0
for text in final_processed_text:
    safety = len(re.findall('safety',text))
    technology = len(re.findall('technology',text))
    crash = len(re.findall('crash',text))
    people = len(re.findall('people',text))
    driver = len(re.findall('driver',text))
    human= len(re.findall('human',text))
    lidar= len(re.findall('lidar',text))
    autonomous= len(re.findall('autonomous',text))
    research = len(re.findall('research',text))
    selfdriving  = len(re.findall('selfdriving ',text))
    policy  = len(re.findall('policy ',text))
    standard  = len(re.findall('standard ',text))
    rule  = len(re.findall('rule ',text))
    government = len(re.findall('government ',text))
    testing = len(re.findall('testing ',text))
    electric  = len(re.findall('electric ',text))
    engineer  = len(re.findall('engineer ',text))
    system  = len(re.findall('system ',text))    
    
    s = safety + crash + people + driver + human + policy + standard + rule + government + testing
    t = technology + lidar + autonomous + research + selfdriving + electric + engineer +  system
    if s > t: 
       safetyCount +=1
    elif s < t:
       techCount += 1
    elif s == t:
        # if equal becomes safety
        safetyCount +=1
    #print('Doc index[' + str(counter) + '], safety: ' + str(s) + ', technology: '+ str(t))
    counter +=1
    
print('Safety Count:' + str(safetyCount) + ', Technology Count:' + str(techCount) + ', Equal Count:' + str(equal))


manualclassTermsEquivalents = [['safety', 'crash','people','driver','human','policy', 'standard', 'rule','government',\
                                'testing'],
                               ['technology','lidar','autonomous','research','selfdriving','electric','engineer','system',\
                                '','']]
manualLabels = pd.DataFrame(data =manualclassTermsEquivalents, index=['safety','technology'],\
                            columns=['Manually','Derived','Equivalent','Terms','','','','','',''])
manualLabels.index.name ='Class Terms'
manualLabels

print('\nManually derived Class Terms and the derived Equivalent Classes from the data above.\n')
print(manualLabels)