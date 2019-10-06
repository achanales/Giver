import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
np.random.seed(400)

def compute_similarity_output_n(art_emb,charity_docs_emb,topn):
    'Compute similarity of article embedding to all charity mission statments and return top N scores and indices in charity df'
    
    #compute cosine distance from article embedding to all charities
    sim_to_charities = cosine_similarity(art_emb,charity_docs_emb)
    
    #find topN similarity scores
    sim_scores_sorted = -np.sort(-sim_to_charities).flatten()
    topN_scores = sim_scores_sorted[:topn]
    
    #find topN indices
    indices_sorted = (-sim_to_charities).argsort().flatten()
    topN_indices = indices_sorted[:topn].flatten()
    
    return topN_scores, topN_indices


def topN_ranked_charities(charity_df, topN_scores, topN_indices):
    'Return df of top N ranked charities'
    
    # Trim charity df
    charity_df_slim = charity_df[['name','category','subcategory','score','description']]
    
    #Extract topN charities and info
    similar_charities = charity_df_slim.iloc[topN_indices].reset_index(drop=True)
    
    #Add their similarity scores
    similar_charities['sim_score'] = topN_scores
    
    return similar_charities

