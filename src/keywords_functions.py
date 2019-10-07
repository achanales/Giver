import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
np.random.seed(400)

#NLP packages
from gensim.corpora import Dictionary
from gensim.matutils import sparse2full
import spacy
nlp  = spacy.load('en_core_web_md')


def find_keywords_df(df,news_tfidf):
    'Find top n keywords for df of articles'
    
    article_tfidf = tfidf_article(df['content'],news_tfidf['news_dict'],news_tfidf['news_tfidf_model'])
    
    df['keywords'] = find_keywords(article_tfidf,news_tfidf['news_dict'],n=10)
    
    return df



def find_keywords(article_tfidf,news_dict,n=10):
    'Funciton to find top n keywords in news article based on tfidf'
    
    
    #Sort words from article by tfidf values
    s = sorted(article_tfidf,key=get_key,reverse=True)
    
    # If there are fewer words than requested keywoards return empty string, if not return keywords
    if len(s) < n:
        return [' ']
    else:
        # Return indices of top N words with highest tfidf values
        topn_tfidf_values = [s[i][0] for i in range(0,n)]
        
        # Look up words in article
        topn_words = [news_dict[i] for i in topn_tfidf_values]
        
        
        return topn_words

def get_key(item):
    return item[1]


def tfidf_article(text,news_dict,news_tfidf_model):
    'Convert article text into TFIDF vector'
    
    text_pre = preprocess_spacy(text)
    
    #Convert docs into tf-idf vectors
    doc_corpus = news_dict.doc2bow(text_pre)
    doc_tfidf  = news_tfidf_model[doc_corpus]
    
    
    return doc_tfidf

def preprocess_spacy(raw_text):
    'Takes raw text, removes unwanted entities and stopwords, lemmatizes, and tokenizes. Returns list of word tokens'
    doc = nlp(raw_text)
    
    # Remove organizations, people, date, and money entities from document text
    tokens_ner = [entity.text for entity in doc.ents if entity.label_ in {'DATE', 'PERSON', 'ORG', 'MONEY', 'GPE'}]
    
    for term in tokens_ner:
        raw_text = raw_text.replace(term, "")
    
    # Re-convert preprocessed text to spacy object
    doc = nlp(raw_text)

    # Remove stopwords and lemmatize
    tokens = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct or token.shape_ == 'dd')]
    
    return tokens
