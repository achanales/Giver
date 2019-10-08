import json
import pandas as pd
import numpy as np
import pickle
from flask import escape
import requests
from google.cloud import storage
import zipfile
import os
import shutil

# NLP Packages
import spacy
nlp  = spacy.load('en_core_web_sm')

from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import sparse2full

def return_ranked_charities(request):
    'Handler function that recieves HTTP request'

    ##### Exctract headline and article #####
    request_json = request.get_json(silent=True)
    
    headline = request_json['headline']
    article = request_json['article']
    
        
   
    
    #### Extract keywoards from article ####
    # Download news tfidf model
   
    download_blob('charity_recommender', 'news_tfidf_min_300_max_0.2.pickle', '/tmp/news_tfidf_min_300_max_0.2.pickle')
    with open('/tmp/news_tfidf_min_300_max_0.2.pickle', 'rb') as handle:
        news_tfidf = pickle.load(handle)
    
    print('Finding keywords')
    # Preprocess artcile
    article_pre = preprocess_spacy(article,nlp)
    
    # Compute tfidf values for each word in article
    article_tfidf = tfidf_article(article_pre,news_tfidf['news_dict'],news_tfidf['news_tfidf_model'])
    
    # Find top 10 Keywords
    keywords = find_keywords(article_tfidf,news_tfidf['news_dict'],n=10)
    print('Keywords found')
    print(keywords)
    
    ### Preprocess and Embed Headline + Keywords ########
    print('Preprocessing headline + keywoard')
    
    headline_keywords = headline + ' '  + ', '.join(keywords)
    print(headline_keywords)
    
    # Preprocess headline + keywords
    headline_keywords_pre = preprocess_spacy(headline_keywords,nlp)
    print('Preprocessed headline + keywoard')

    download_blob('charity_recommender', 'charity_model_min_0_max_0.5_notfidf.pickle', '/tmp/charity_model_min_0_max_0.5_notfidf.pickle')
    download_blob('charity_recommender', 'charity_data_cleaned.csv', '/tmp/charity_data_cleaned.csv')
    
    # Load Charity dataset
    file_name = '/tmp/charity_data_cleaned.csv'
    all_charity = pd.read_csv(file_name)

    # Load Trained Charity Model
    with open('/tmp/charity_model_min_0_max_0.5_notfidf.pickle', 'rb') as handle:
        charity_model = pickle.load(handle)
    

    # Unpack model variables from pickle
    charity_docs_dict = charity_model['charity_docs_dict']
    charity_emb_vecs =  charity_model['charity_emb_vecs']
    charity_docs_emb =  charity_model['charity_docs_emb']
    
    # Clean up workspace to free up memory
    del charity_model
    os.remove('/tmp/charity_model_min_0_max_0.5_notfidf.pickle')
    os.remove('/tmp/charity_data_cleaned.csv')
    
    
    # embed in document vector
    headline_keywords_emb = embed_text(headline_keywords_pre,charity_docs_dict,charity_emb_vecs)
    
    # Compute the similarity of headline to all mission statements and return top 3 ranked charities scores and indices
    topN_scores, topN_indices = compute_similarity_output_n(headline_keywords_emb,charity_docs_emb,3)
	 
    # Lookup top charities in charity database and retreive relevant information
    topN_charities = topN_ranked_charities(all_charity, topN_scores, topN_indices)
    
    topN_charities['link'] = 'www.google.com/search?q=' + topN_charities['name'].apply(lambda x: x.replace(' ','+'))
    topN_charities['keywords'] = ', '.join(keywords)
	
    # Convert to json
    return_json = topN_charities.to_json(orient='index')
    
    # Output JSON file of top ranked charities including ['name','category','subcategory','score','description', 'sim_score']
    return return_json
    


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    
    blob.download_to_filename(destination_file_name)
    

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))

    
def preprocess_spacy(raw_text,nlp):

    'Function that takes a string of text as input and returns a list of preprocessed tokens'

    doc = nlp(raw_text)
    
    #Remove organizations and people from documnet text
    tokens_ner = [entity.text for entity in doc.ents if entity.label_ in {'DATE', 'PERSON', 'ORG','MONEY','GPE'}]

    for term in tokens_ner:
        raw_text = raw_text.replace(term,"")
    
    #Re-convert preprocessed text to spacy object    
    doc = nlp(raw_text)


    #Remove stopwords and lemmatize
    tokens = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct or token.shape_ == 'dd')]
    return tokens

def tokenize_article(raw_text,nlp):

    'Function that takes a string of text and returns word tokens lemmatized'

    doc = nlp(raw_text)
 
    #Remove stopwords and lemmatize
    tokens = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct or token.shape_ == 'dd')]
    return tokens

def embed_text(text_pre,charity_docs_dict,charity_emb_vecs):

    'Function that takes a string of text as input, preprocesses and tokenizes the text and embeds into a sentence vector'
     
    #Convert docs into tf-idf vectors
    doc_corpus = charity_docs_dict.doc2bow(text_pre)
    doc_vec   = np.vstack([sparse2full(doc_corpus, len(charity_docs_dict))])
    
    # sum of glove vectors linearlly weighted by tfidf 
    headline_emb = np.dot(doc_vec, charity_emb_vecs)
    
    
    return headline_emb

def compute_similarity_output_n(headline_emb,charity_docs_emb,topn):
    
    'Function that computes the cosine similarity between the headline vector and all mission statement vectors and returns topN similarity scores and their indices to be used to lookup charities in charity db'


    #compute cosine distance from article embedding to all charities
    sim_to_charities = cosine_similarity(headline_emb,charity_docs_emb)
    
    #find topN similarity scores
    sim_scores_sorted = -np.sort(-sim_to_charities).flatten()
    topN_scores = sim_scores_sorted[:topn]
    
    #find topN indices
    indices_sorted = (-sim_to_charities).argsort().flatten()
    topN_indices = indices_sorted[:topn].flatten()
    
    return topN_scores, topN_indices

def cosine_similarity(vecA,matB):
    cos_all = []

    for row in matB:
        dot = np.dot(vecA, row)
        norma = np.linalg.norm(vecA)
        normb = np.linalg.norm(row)
        cos = dot / (norma * normb)
        cos_all.append(cos[0])
    return np.asarray(cos_all)

def topN_ranked_charities(charity_df, topN_scores, topN_indices):
    
    'Returns the topN charities to headline'

    charity_df_slim = charity_df[['name','category','subcategory','score','description']]
    
    #Extract topN charities and info
    similar_charities = charity_df_slim.iloc[topN_indices].reset_index(drop=True)
    
    #Add their similarity scores
    similar_charities['sim_score'] = topN_scores
    
    return similar_charities

def tfidf_article(text_pre,news_dict,news_tfidf_model):
         
    #Convert docs into tf-idf vectors
    doc_corpus = news_dict.doc2bow(text_pre)
    doc_tfidf  = news_tfidf_model[doc_corpus]
    
    
    return doc_tfidf

def getKey(item):
    return item[1]

def find_keywords(article_tfidf,news_dict,n=10):
    'Funciton to find top n keywords in news article based on tfidf'
    
    #Sort words from article by tfidf values
    s = sorted(article_tfidf,key=getKey,reverse=True)
    
    #Check if there are fewer than number of keyword in article
    if len(s) < n:
        return [' ']
    else: 
        # Return indices of top N words with highest tfidf values
        topn_tfidf_values = [s[i][0] for i in range(0,n)]
    
        # Look up words in article
        topn_words = [news_dict[i] for i in topn_tfidf_values]
        
        return topn_words

