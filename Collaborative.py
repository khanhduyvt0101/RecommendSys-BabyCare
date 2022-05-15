import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


class Collaborative(object):
    # Neighborhood-based Collaborative Filtering (NBCF)
    def __init__(self, Y_data, k, dist_func=cosine_similarity, uuCF=1):
        self.uuCF = uuCF  # user-user Collaborative Filtering
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.k = k
        self.dist_func = dist_func
        self.Ybar_data = None
        # Number of users and hotel.
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    # Update Y_data with new data
    def addNewData(self, newRating):
        self.Y_data = np.concatenate((self.Y_data, newRating), axis=0)

    # Normalize
    def normalize_Y(self):
        users = self.Y_data[:, 0]  # Get users
        self.Ybar_data = self.Y_data.copy()  # Make a copy of Y_data
        self.utilityMatrix = np.zeros((self.n_users,))
        for n in range(self.n_users):
            # rating index that was voted by user n
            ids = np.where(users == n)[0].astype(np.int32)
            # hotelId that rating by user n
            item_ids = self.Y_data[ids, 1]
            # and the rating
            ratings = self.Y_data[ids, 2]
            # Calc mean of rating
            mean = np.mean(ratings)
            if np.isnan(mean):
                mean = 0  # to avoid empty array and NaN value
            self.utilityMatrix[n] = mean
            # Normalize
            self.Ybar_data[ids, 2] = ratings - self.utilityMatrix[n]
            print(type(self.Ybar_data[ids, 2]))

        # create -sparse matrix
        # -> just save values which != 0 and the position
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
                                       (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        # print(self.Ybar)
        self.Ybar = self.Ybar.tocsr()

    # Similarity function
    def similarity(self):
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)

    # Training function
    def training(self):
        # Normalize data and calc sim_matrix after add ratings
        self.normalize_Y()
        self.similarity()

    # Rating Prediction function
    def pred(self, u, i, normalized=1):
        # Predict the rating of user u for item i (normalized)

        # Find all users that rated i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)

        # Find similarities between current user and others
        sim = self.S[u, users_rated_i]

        # Find k most similar users
        a = np.argsort(sim)[-self.k:]
        # And the degree of similarity
        nearest_s = sim[a]
        r = self.Ybar[i, users_rated_i[a]]
        if normalized:
            # Add a small number to avoid division by 0
            return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8)

        return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8) + self.utilityMatrix[u]

    # Rating Prediction function for uuCF and iiCF
    def predict(self, user, item, normalized=1):
        if self.uuCF:
            return self.pred(user, item, normalized)
        return self.pred(item, user, normalized)

    # Recommend item
    def recommendItems(self, user):
        # Identify all items that should be recommended to user.
        #  Based on : self.pred(u, i) > 0 -> Assume looking at items not yet rated by user
        ids = np.where(self.Y_data[:, 0] == user)[0]
        items_rated_by_user = self.Y_data[ids, 1].tolist()
        recommended_items = []  # List recommended items
        for i in range(self.n_items):
            if i not in items_rated_by_user:
                rating = self.pred(user, i)
                if rating > 0:
                    recommended_items.append([i, rating])
        # Sorting
        recommended_items.sort(reverse=True, key=lambda x: x[1])

        return recommended_items

    # Get all items which should be recommended for each user
    def get_recommendation(self, userId):
        recommended_items = self.recommendItems(userId)
        return recommended_items

    # Get RMSE
    def getRMSE(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0  # squared error
        for index, row in rate_test.iterrows():
            u_id = row[0]
            i_id = row[1]
            point = row[2]
            pred = self.predict(u_id, i_id, normalized=0)
            SE += (pred - point) ** 2

        RMSE = np.sqrt(SE / n_tests)
        return RMSE
