# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 09:34:29 2023

@author: hoang
"""

import pandas as pd
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


nltk.download('punkt')

# Import the dataset
df = pd.read_csv('Datasets/cleaned_data.csv')
df.info()

# Create an English language SnowballStemmer object
stemmer = SnowballStemmer('english')

# Define a function to perform both stemming and tokenization
def tokenize_and_stem(text):
    # Tokenize by sentence, then by word
    tokens = [word for sent in nltk.sent_tokenize(text) 
                  for word in nltk.word_tokenize(sent)]
    
    # Filter out raw tokens to remove noise
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
    
    # Stem the filterd tokens
    stem = [stemmer.stem(word) for word in filtered_tokens]
    
    return stem
    
# Transform token into features using TF-IDF
    # Instantiate TfidfVectorizer object with stopwords and tokenizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0.2,
                                   stop_words='english', use_idf=True,
                                   tokenizer=tokenize_and_stem,
                                   ngram_range=(1,3))

# Fit and transform the tfidf_vectorizer on the Gameplay column
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Gameplay'])

# ---------------------- KMeans ------------------------------
from sklearn.cluster import KMeans  
from sklearn.metrics import silhouette_score

# --- Elbow Method and Silhouette Method
n_clusters = []
sse_list = []
silhouette_scores = []
for k in range(2, 15):
    # Initiate KMeans object and fit and predict the tfidf_matrix
    km = KMeans(n_clusters=k)
    preds = km.fit_predict(tfidf_matrix)
    
    # Calculate SSE using inertia_ attribue, 
    # which is the sum of the squared distances of each point to its closest centroid
    sse = km.inertia_
    
    # Find the average silhouette score
    ss = silhouette_score(tfidf_matrix, preds)
    
    # Add k and see to n_cluster, sse_lists and silhouette_scores respectively
    n_clusters.append(k)
    sse_list.append(sse)
    silhouette_scores.append(ss)
    
# Plot the SSE score 
fig = plt.figure(figsize=(15,8))
fig.add_subplot(121)
plt.plot(n_clusters, sse_list, 'd-', label='Sum of squared error')
plt.xlabel('Number of cluster')
plt.ylabel('SSE')
plt.legend()
plt.show()
    # There is no clear elbow point for KMeans clustering.
    
# Plot the Silhouette scores
fig.add_subplot(122)
plt.plot(n_clusters, silhouette_scores, 'd-', label='Silhouette Score')
plt.xlabel('Number of cluster')
plt.ylabel('Silhouette score')
plt.legend()
plt.show()
    # The average silhouette score is pretty close to zero 
    # which means the sample is close to the decision boundary between two neighboring clusters

# Combining the two methods, we can choose k = 11

kmeans = KMeans(n_clusters = 11)
kmeans.fit(tfidf_matrix)
clusters = kmeans.labels_.tolist()
df['Cluster'] = clusters
df['Cluster'].value_counts()


# ------------------------- Hierarchy -------------------------------
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram

# Calculate the similarity distance
sim_distance = 1 - cosine_similarity(tfidf_matrix)

# Create merging matrix
merging_matrix = linkage(sim_distance, method='complete')

# Plot the dendrogram using title as label column
dendrogram_plot = dendrogram(merging_matrix,
                             labels=df['Title'].tolist(),
                             leaf_rotation=90,
                             leaf_font_size=16)

fig = plt.gcf()
_ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
fig.set_size_inches(108, 21)
# plt.savefig('dendrogram.png', dpi=100)
plt.show()


# Create a dataframe from the similarity matrix
title = df['Title'].tolist()
sim_df = pd.DataFrame(sim_distance, columns=title, index=title)
# sim_df.to_csv('Datasets/similarity_matrix.csv')

# Recommendation Example

# game_title = "Pokémon: Let's Go, Pikachu!"
game_title = "The Legend of Zelda: Tears of the Kingdom"

# Find 5 matching games that have smallest similarity distance to the chosen game.
matches = sim_df[game_title].sort_values().drop(index=game_title).head(5)
matching_games = matches.index.tolist()
# Extract the information of those matching games
matching_games_df = df.set_index('Title').loc[matching_games]





