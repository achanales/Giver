from sklearn.cluster import KMeans

def predict_kmeans(model,doc_emb):
    
    prediction = model.predict(art_emb)
    
    return prediction