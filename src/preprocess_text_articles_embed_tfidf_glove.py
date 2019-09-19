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


def word_embed_charity(processed_text,doc_dict,tfidf_emb_vecs):
    
    doc_bow = docs_dict.doc2bow(processed_text)
    
    doc_tfidf = model_tfidf[doc_bow]
    
    doc_vec = np.vstack([sparse2full(a, len(docs_dict))])
    
    # sum of glove vectors linearlly weighted by tfidf 
    art_emb = np.dot(docs_vecs, tfidf_emb_vecs)
    
    return art_emb

