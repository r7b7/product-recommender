import pandas as pd

from surprise import Dataset, NormalPredictor, Reader, SVD, accuracy
from surprise.model_selection import cross_validate, train_test_split


class SVDModel:
    def __init__(self):
        self.model = None

    def train_model(self):
        # Fetch built-in movie ratings dataset
        data = Dataset.load_builtin('ml-100k')
        trainset = data.build_full_trainset()
        self.model = SVD()
        self.model.fit(trainset)
        print('Training done')

    def predict_rating(self, user_id, item_id):
        prediction = self.model.predict(user_id, item_id, r_ui=None, verbose=True)
        return prediction.est