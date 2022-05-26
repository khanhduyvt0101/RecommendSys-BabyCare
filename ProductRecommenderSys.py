import ReadData as fsData
import numpy as np
import pandas as pd
import re
from tabulate import tabulate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import Collaborative as CF

products_data = fsData.getProductsData()
ratings_data = fsData.getRatingData()

C = products_data['ratePoint'].mean()  # C is the mean vote across the whole report.
m = products_data['rateCount'].quantile(0.75)  # m is the minimum votes required to be listed in the chart.

# Filter out all qualified products into a new DataFrame
q_products = products_data.copy().loc[products_data['rateCount'] >= m]


# ================================ #

# Function that computes the weighted rating of each product
def weighted_rating(row, m=m, C=C):
    v = int(row['rateCount'])
    R = int(row['ratePoint'])
    return (v / (v + m) * R) + (m / (m + v) * C)


# Merge column values into one string and remove unuseful characters
def merge_info(row):
    baseString = str(row['name'] + ' ' + row['shopLocation'] + ' ' + row['tagName'] + '  ' + row['type'])
    # Remove unuseful words
    # unusefulWords = ['·', 'Vietnam', 'Việt Nam', ',', 'Phòng ', 'phòng ', '-']
    unusefulWords = []
    big_regex = re.compile('|'.join(map(re.escape, unusefulWords)))
    usefulString = big_regex.sub('', baseString)
    # Combine number and word into one word
    result = re.sub('(?<=\d) (?=\w)', '', usefulString)
    return result


# ========== Constant ============ #
TEST_PERCENT = 0.2  # test 20% of data
K_USER = 100
UUCF = 1


class ProductRecommenderSys(object):
    def __init__(self):
        self.dfProducts = products_data
        self.dfRatings = ratings_data

    # Get recommend products based on WEIGHTED RATING
    def get_weighted_rating(self):
        # Define a new feature 'score' and calculate its value with `weighted_rating()`
        q_products['score'] = q_products.apply(weighted_rating, axis=1)

        # Sort products based on score calculated above
        result = q_products.sort_values('score', ascending=False)

        # Get top 6 highest score
        return result['id'][:7].to_list()

    # Get recommend products based on Content-based Filtering
    def get_content_based(self, id):
        df = self.dfProducts
        # Filter out all qualified products based on city into a new DataFrame
        row_index_matchId = df.index[df['id'] == id].tolist()
        df = df.copy().loc[df['tagName'] == df['tagName'].values[row_index_matchId[0]]]

        # Construct a reverse map of indices and idProduct
        indices = pd.Series(df.index, index=df['id']).drop_duplicates()

        # Get the index of the product that matches the idProduct
        idx = indices[id]

        # Columns
        columns = ['name', 'shopLocation', 'tagName', 'type']

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

        # Get the pairwsie similarity scores of all products with that product
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))

        # Sort the products based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 6 most similar products
        sim_scores = sim_scores[1:7]

        # Get the product indices
        products_indices = [i[0] for i in sim_scores]

        result = df['id'].iloc[products_indices]

        # Return the top 6 most similar products
        return result.to_list()

    # Get recommend products based on Collaborative Filtering
    # def get_collaborative(self, userId):
    #     # prepare data
    #     df = self.dfRatings
    #     dfRating = df.filter(items=['userId', 'hotelId', 'ratingPoint'])
    #     rate_train, rate_test = train_test_split(dfRating, test_size=TEST_PERCENT)

    #     # Training data
    #     Y_data = rate_train.to_numpy()
    #     rs = CF.Collaborative(Y_data, k=K_USER, uuCF=1)
    #     rs.training()

    #     # RMSE on test data
    #     print('RMSE =', rs.getRMSE(rate_test))

    #     return rs.get_recommendation(userId)
