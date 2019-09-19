import pandas as pd;
import numpy as np;
import scipy as sp;
import sklearn;
import sys;
from nltk.corpus import stopwords;
import nltk;
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
from sklearn.decomposition import PCA
from sklearn import manifold
import pickle;


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import sparse2full
from nltk.stem.porter import *
import numpy as np
np.random.seed(400)

nltk.download('stopwords')
stemmer = SnowballStemmer("english")


'''
Write a function to run word embedding pre-processing on df
'''
def word_embed_charity(df,pca_comp =8,word_min=5, word_max_perc=.2 ):
    
    #Extract mission descritptions
    mission_text = df['description'].astype('str')
    
    #Pre-process documents (remove stop words and lamentize)
    processed_docs = []

    for doc in mission_text:
        processed_docs.append(preprocess(doc))
    
    #Create dictionary from corpus
    docs_dict = Dictionary(processed_docs)
    docs_dict.filter_extremes(no_below=word_min, no_above=word_max_perc)
    docs_dict.compactify()
    
    #Convert docs into tf-idf vectors
    docs_corpus = [docs_dict.doc2bow(doc) for doc in processed_docs]
    model_tfidf = TfidfModel(docs_corpus, id2word=docs_dict)
    docs_tfidf  = model_tfidf[docs_corpus]
    docs_vecs   = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_tfidf])
    
    num_docs= np.shape(docs_vecs)[0]
    num_words = np.shape(docs_vecs)[1]

    print("Total # of docs: {}".format(num_docs))
    print("Total # of words in dict: {}".format(num_words))
    
    #For each word in dict obtain embedding vector (Glove vectors)
    tfidf_emb_vecs = np.vstack([nlp(docs_dict[i]).vector for i in range(len(docs_dict))])
    
    # Weight glove vectors by tf-idf values
    docs_emb = np.dot(docs_vecs, tfidf_emb_vecs) 
        
    return docs_emb, docs_dict, model_tfidf, docs_dict


def pca_embed_matrix(df, docs_emb,pca_comp =8):
    
    #Perform pca on document matrix
    docs_pca = PCA(n_components=pca_comp).fit_transform(docs_emb)
    
    #Project into 2D space
    tsne = manifold.TSNE()
    viz = tsne.fit_transform(docs_pca)
    
    return viz
