import warnings
warnings.filterwarnings('ignore')

import numpy as np
np.random.seed(400)

#NLP packages
from gensim.corpora import Dictionary
from gensim.matutils import sparse2full
import spacy
nlp  = spacy.load('en_core_web_md')


##################################
## Preprocessing fucntions ######
##################################


def preprocess_spacy(raw_text):
    'Takes raw text, removes unwanted entities and stopwords, lemmatizes, and tokenizes. Returns list of word tokens'
    doc = nlp(raw_text)

    # Remove organizations, people, date, and money entities from document text
    tokens_ner = [entity.text for entity in doc.ents if entity.label_ in {'DATE', 'PERSON', 'ORG', 'MONEY'}]

    for term in tokens_ner:
        raw_text = raw_text.replace(term, "")

    # Re-convert preprocessed text to spacy object
    doc = nlp(raw_text)

    # Remove stopwords and lemmatize
    tokens = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct or token.shape_ == 'dd')]

    return tokens


def preprocess_docs(docs):
    'Preprocess a list texts'

    result = []

    for doc in docs:
        result.append(preprocess_spacy(doc))


##################################
## Document Embedding fucntions ######
##################################

def doc_embed_charity(processed_docs, word_min=5, word_max_perc=.8):
    'Takes a list of preprocessed texts and returns an embedding vector for each document, a dictionary of the words within the corpus, and the glove vectors for each word in the corpus'

    # Create dictionary from corpus
    docs_dict = Dictionary(processed_docs)
    docs_dict.filter_extremes(no_below=word_min, no_above=word_max_perc)
    docs_dict.compactify()

    # Convert docs into sparce matricx (N_docs x N_words in dictionary) where the number in each cell indicates the number of time that word appeared in that document
    docs_corpus = [docs_dict.doc2bow(doc) for doc in processed_docs]
    docs_vecs = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_corpus])

    #Count number of documents and words in dictionary
    num_docs = np.shape(docs_vecs)[0]
    num_words = np.shape(docs_vecs)[1]

    print("Total # of docs: {}".format(num_docs))
    print("Total # of words in dict: {}".format(num_words))

    # For each word in dict extract embedding vector (Glove vectors)
    glove_vecs = np.vstack([nlp(docs_dict[i]).vector for i in range(len(docs_dict))])

    # Sum glove vectors over words in doc
    docs_emb = np.dot(docs_vecs, glove_vecs)

    return docs_emb, docs_dict, glove_vecs


def doc_embed_article(processed_text, docs_dict, glove_vecs):

    'Filters input text to only include words in input dictionary and then embeds input text into vector'

    # Filter words to only include the ones in th input dictionary
    doc_corpus = docs_dict.doc2bow(processed_text)
    doc_vec = np.vstack([sparse2full(doc_corpus, len(docs_dict))])

    # sum of glove vectors over words in doc
    doc_emb = np.dot(doc_vec, glove_vecs)

    return doc_emb


