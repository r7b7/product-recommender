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
    "1": "Eat Pray Love",
    "2": "Fast and Furious1",
    "3": "Chocolat",
    "4": "Fast and Furious2"
    }
    contentRecommendations = ContentBasedRecommender()
    recommended_movie_ids = contentRecommendations.get_recommendations(user_id)
    print('recommended ids', recommended_movie_ids)

    recommended_movies = []
    for movie_id in recommended_movie_ids:
        rating = svd_model.predict_rating(user_id, movie_id)
        print('rating is', rating)
        if rating > 3:
            recommended_movies.append(movies[movie_id])

    return {"recommended_movies": recommended_movies}