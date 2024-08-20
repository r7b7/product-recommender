from fastapi import FastAPI
from models.svd_trained import SVDModel
from recommender import ContentBasedRecommender
from contextlib import asynccontextmanager

svd_model = SVDModel()

@asynccontextmanager
async def lifespan(app: FastAPI):
    svd_model.train_model()
    yield
    print("Cleaning up")

recommendApp = FastAPI(lifespan=lifespan)

@recommendApp.get("/recommendation/{user_id}")
def get_recommendation(user_id: str):
    movies = {
    "61": "Eat Pray Love",
    "302": "Fast and Furious1",
    "31": "Chocolat",
    "311": "Fast and Furious2"
    }
    contentRecommendations = ContentBasedRecommender()
    recommended_movie_ids = contentRecommendations.get_recommendations(user_id)
    history_based_recommendation = []
    recommended_movies = []

    for movie_id in recommended_movie_ids:
        history_based_recommendation.append(movies[movie_id])

    for movie_id in movies.keys():
        rating = svd_model.predict_rating(user_id, movie_id)
        if rating > 3.90:
            recommended_movies.append(movies[movie_id])

    return {"recommended_movies": recommended_movies, "history_based_recommendation": history_based_recommendation}