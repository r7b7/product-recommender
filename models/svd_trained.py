import pandas as pd

from surprise import Dataset, NormalPredictor, Reader, SVD, accuracy
from surprise.model_selection import cross_validate, train_test_split


class SVDModel:
    def __init__(self):
        self.model = None

    def train_model(self):
        # Synthetic ratings data
        ratings_dict = {
            "itemID": [1, 1, 1, 2, 2, 2, 3, 3, 4],
            "userID": [1, 2, 4, 2, 3, 4, 1, 4, 3],
            "rating": [4, 4, 4, 3, 4, 1, 4, 4, 4],
        }
        df = pd.DataFrame(ratings_dict)
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[["itemID", "userID", "rating"]], reader)
        trainset, testset = train_test_split(data, test_size=0.25)
        self.model = SVD()
        self.model.fit(trainset)
        print('Training done')

    def predict_rating(self, user_id, item_id):
        prediction = self.model.predict(user_id, item_id, r_ui=None, verbose=True)
        return prediction.est