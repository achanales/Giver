import json
import boto3

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# NLP Packages
import spacy
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import sparse2full
import pdb

nlp  = spacy.load('en_core_web_md')


def lambda_handler(event, context):

	headline = event['headline']

	with open('/Users/avi/Dropbox/Insight/Project/insight_project/data/processed/charity_model_min_0_max_0.5.pickle', 'rb') as handle:
		charity_model = pickle.load(handle)

	file_name = '/Users/avi/Dropbox/Insight/Project/insight_project/data/interim/charity_data_cleaned.csv'
	all_charity = pd.read_csv(file_name)

	charity_docs_dict = charity_model['charity_docs_dict']
	charity_model_tfidf = charity_model['charity_model_tfidf']
	charity_tfidf_emb_vecs =  charity_model['charity_tfidf_emb_vecs']
	charity_docs_emb =  charity_model['charity_docs_emb']

	headline_emb = process_embed_text(headline,charity_docs_dict,charity_model_tfidf,charity_tfidf_emb_vecs)
	topN_scores, topN_indices = compute_similarity_output_n(headline_emb,charity_docs_emb,3)
	topN_charities = topN_ranked_charities(all_charity, topN_scores, topN_indices)

	return {
        'statusCode': 200,
        'body': topN_charities.to_json(orient='index')
    }

def preprocess_spacy(raw_text):

    'Function that takes a string of text as input and returns a list of preprocessed tokens'

    doc = nlp(raw_text)
    
    #Remove dates, organizations and people from documnet text
    tokens_ner = [entity.text for entity in doc.ents if entity.label_ in {'DATE', 'PERSON', 'ORG'}]

    for term in tokens_ner:
        raw_text = raw_text.replace(term,"")
    
    #Re-convert preprocessed text to spacy object    
    doc = nlp(raw_text)


    #Remove stopwords, punctuation and lemmatize
    tokens = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct)]

    return tokens

def process_embed_text(text,charity_docs_dict,charity_model_tfidf,charity_tfidf_emb_vecs):

    'Function that takes a string of text as input, preprocesses and tokenizes the text and embeds into a sentence vector'

    text_pre = preprocess_spacy(text)
     
    #Convert docs into tf-idf vectors
    doc_corpus = charity_docs_dict.doc2bow(text_pre)
    doc_tfidf  = charity_model_tfidf[doc_corpus]
    doc_vec   = np.vstack([sparse2full(doc_tfidf, len(charity_docs_dict))])
    
    # sum of glove vectors linearlly weighted by tfidf 
    headline_emb = np.dot(doc_vec, charity_tfidf_emb_vecs)
    
    
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

def topN_ranked_charities(charity_df, topN_scores, topN_indices):
    
    'Returns the topN charities to headline'

    charity_df_slim = charity_df[['name','category','subcategory','score','description']]
    
    #Extract topN charities and info
    similar_charities = charity_df_slim.iloc[topN_indices].reset_index(drop=True)
    
    #Add their similarity scores
    similar_charities['sim_score'] = topN_scores
    
    return similar_charities

