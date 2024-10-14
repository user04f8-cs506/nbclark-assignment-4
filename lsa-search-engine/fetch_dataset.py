# fetch_dataset.py
import json
import gzip
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups

CHUNK_SIZE = 100

print('Loading dataset. . .')

newsgroups = fetch_20newsgroups(subset='all')
texts = newsgroups.data

print('Loaded! Preprocessing. . .')

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(texts)

svd = TruncatedSVD(n_components=100)
U = svd.fit_transform(X)
S = svd.singular_values_
Vt = svd.components_

data = {
    'S': S.tolist(),
    'Vt': Vt.tolist(),
    'terms': vectorizer.get_feature_names_out().tolist(),
}

with open('public/lsa_metadata.json', 'w') as f:
    json.dump(data, f)

num_chunks = int(np.ceil(len(texts) / CHUNK_SIZE))

for i in range(num_chunks):
    print(f'Saving chunk {i+1}/{num_chunks}. . .')
    start_idx = i * CHUNK_SIZE
    end_idx = start_idx + CHUNK_SIZE
    chunk_U = U[start_idx:end_idx]
    chunk_documents = texts[start_idx:end_idx]
    
    chunk_data = {
        'U': chunk_U.tolist(),
        'documents': [{'text': text} for text in chunk_documents]
    }
    
    # Save the chunk without compression for now
    with open(f'public/chunks/chunk_{i}.json', 'w', encoding='utf-8') as json_file:
        json.dump(chunk_data, json_file)
        
    # with gzip.open(f'public/chunks/chunk_{i}.json.gz', 'wt', encoding='utf-8') as gz_file:
    #     json.dump(chunk_data, gz_file)
