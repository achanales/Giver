from sklearn.cluster import KMeans

def run_kmeans(n_clusters,docs_emb,max_iter = 500, n_init=15):
    
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=max_iter, n_init=n_init)
    model.fit(charity_docs_emb)
    
    return model