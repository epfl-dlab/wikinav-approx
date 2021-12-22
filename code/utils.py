import os
import csv
import pickle
import numpy as np
from numpy.linalg import norm
import gensim
from gensim.models.keyedvectors import KeyedVectors

def loadWordVectors(path):
    if "bin" in path:
        isBinary = True
    else:
        isBinary = False
    word_vectors = KeyedVectors.load_word2vec_format(path, binary=isBinary)
    return word_vectors

def mapId2Name(PATH_IN, month):
    name2id = {}; id2name = {}; is_redirect = {}
    with open(os.path.join(PATH_IN, f'page_{month}.csv')) as fpage:
        reader = csv.reader(fpage, delimiter=',', quotechar='"')
        lnum = 0
        for line in reader:
            if lnum == 0: # Skip header row
                lnum+=1
                continue
            pid = int(line[0]); ptitle = line[1]; isRedirect = line[2]
            id2name[pid] = ptitle
            if isRedirect == "False":
                is_redirect[pid] = False
            else:
                is_redirect[pid] = True
            name2id[ptitle] = pid

            lnum+=1
    return id2name, name2id, is_redirect

def get_sentence_embedding(words, model):
    embeddings = []
    for word in words:
        try:
            word_emb = model[word]
            if norm(word_emb) != 0:
                embeddings.append((word_emb/norm(word_emb)).tolist())
        except KeyError:
            continue
    embeddings = np.array(embeddings, dtype='float64')
    return embeddings

def get_sentence_embedding_xlm(sentences, model):
    embedding_array = []
    embeddings = model.encode(sentences)
    for embedding in embeddings:
        embedding_array.append((embedding/norm(embedding)).tolist())
    embedding_array = np.array(embedding_array, dtype='float64')
    return embedding_array

def get_description_embeddings(text_tokens_processed, text_tokens, embeddingType, emb_models):
    if embeddingType in ['w2v', 'glove', 'fasttext']:
        embedding_array = get_sentence_embedding(text_tokens_processed, emb_models[embeddingType])
    else:
        embedding_array = get_sentence_embedding_xlm(text_tokens, emb_models[embeddingType])
    return embedding_array
