import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class ContentBasedRecommender:
    def __init__(self):
        pass

    def get_recommendations(self, user: str):
        #fetch all movies from DB
        movies = pd.DataFrame({
            'title-id': ['1', '2', '3', '4'],
            'genres': ['Rom-Com', 'Action', 'Rom-Com', 'Action']
        })

        #fetch movies watched by user from DB
        user_movies = ['1']

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies['genres'])

        user_profile = tfidf_matrix[movies['title-id'].isin(user_movies)].mean(axis=0)
        similarity_scores = cosine_similarity(np.asarray(user_profile), tfidf_matrix)
        movies['similarity'] = similarity_scores.flatten()
        recommended_movies = movies[~movies['title-id'].isin(user_movies)].sort_values(by='similarity', ascending=False)
        return recommended_movies['title-id'].tolist()