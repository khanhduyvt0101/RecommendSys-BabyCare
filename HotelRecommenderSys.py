import ReadData as fsData
import numpy as np
import pandas as pd
import re
from tabulate import tabulate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import Collaborative as CF

hotels_data = fsData.getHotelsData()
ratings_data = fsData.getRatingData()

C = hotels_data['rating'].mean()  # C is the mean vote across the whole report.
m = hotels_data['ratingCount'].quantile(0.75)  # m is the minimum votes required to be listed in the chart.

# Filter out all qualified hotels into a new DataFrame
q_hotels = hotels_data.copy().loc[hotels_data['ratingCount'] >= m]


# ================================ #

# Function that computes the weighted rating of each hotel
def weighted_rating(row, m=m, C=C):
    v = int(row['ratingCount'])
    R = int(row['rating'])
    return (v / (v + m) * R) + (m / (m + v) * C)


# Merge column values into one string and remove unuseful characters
def merge_info(row):
    baseString = str(row['address'] + ' ' + row['type'] + ' ' + row['detailType'] + '  ' + row['detailRoom'])
    # Remove unuseful words
    unusefulWords = ['·', 'Vietnam', 'Việt Nam', ',', 'Phòng ', 'phòng ', '-']
    big_regex = re.compile('|'.join(map(re.escape, unusefulWords)))
    usefulString = big_regex.sub('', baseString)
    # Combine number and word into one word
    result = re.sub('(?<=\d) (?=\w)', '', usefulString)
    return result


# ========== Constant ============ #
TEST_PERCENT = 0.2  # test 20% of data
K_USER = 100
UUCF = 1


class HotelRecommenderSys(object):
    def __init__(self):
        self.dfHotels = hotels_data
        self.dfRatings = ratings_data

    # Get recommend hotels based on WEIGHTED RATING
    def get_weighted_rating(self):
        # Define a new feature 'score' and calculate its value with `weighted_rating()`
        q_hotels['score'] = q_hotels.apply(weighted_rating, axis=1)

        # Sort hotels based on score calculated above
        result = q_hotels.sort_values('score', ascending=False)

        # Get top 6 highest score
        return result['hotelId'][:7].to_list()

    # Get recommend hotels based on Content-based Filtering
    def get_content_based(self, hotelId):
        df = self.dfHotels
        # Filter out all qualified hotels based on city into a new DataFrame
        row_index_matchId = df.index[df['hotelId'] == hotelId].tolist()
        df = df.copy().loc[df['cityId'] == df['cityId'].values[row_index_matchId[0]]]

        # Construct a reverse map of indices and hotelIds
        indices = pd.Series(df.index, index=df['hotelId']).drop_duplicates()

        # Get the index of the hotel that matches the hotelId
        idx = indices[hotelId]

        # Columns
        columns = ['address', 'type', 'detailType', 'detailRoom']

        # Convert column values to String
        for column in columns:
            df[column] = df[column].apply(str)

        # Create new feature mergeInfo by apply merge_info method
        df['mergeInfo'] = df.apply(merge_info, axis=1)

        # Define a TF-IDF Vectorizer Object.
        tfidf = TfidfVectorizer()
        # Construct the required TF-IDF matrix by fitting and transforming the data
        tfidf_matrix = tfidf.fit_transform(df['mergeInfo'])

        # Compute the cosine similarity matrix
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Get the pairwsie similarity scores of all hotels with that hotel
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))

        # Sort the hotels based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 6 most similar hotels
        sim_scores = sim_scores[1:7]

        # Get the hotel indices
        hotel_indices = [i[0] for i in sim_scores]

        result = df['hotelId'].iloc[hotel_indices]

        # Return the top 6 most similar hotels
        return result.to_list()

    # Get recommend hotels based on Collaborative Filtering
    def get_collaborative(self, userId):
        # prepare data
        df = self.dfRatings
        dfRating = df.filter(items=['userId', 'hotelId', 'ratingPoint'])
        rate_train, rate_test = train_test_split(dfRating, test_size=TEST_PERCENT)

        # Training data
        Y_data = rate_train.to_numpy()
        rs = CF.Collaborative(Y_data, k=K_USER, uuCF=1)
        rs.training()

        # RMSE on test data
        print('RMSE =', rs.getRMSE(rate_test))

        return rs.get_recommendation(userId)
