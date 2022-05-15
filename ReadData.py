import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Use a service account
cred = credentials.Certificate('firebase_key/service_account_key.json')
firebase_admin.initialize_app(cred)

db = firestore.client()


# Get Data from firebase
def getHotelsData():
    hotels = list(db.collection(u'hotels').stream())

    hotels_dict = list(map(lambda x: x.to_dict(), hotels))
    df = pd.DataFrame(hotels_dict)

    df['ratingCount'] = df['ratingList'].str.len()  # Create new feature Rating Count
    # Drop unuseful feature
    df.drop(columns=['thumbnail', 'img', 'detailInfo', 'ratingList', 'utilities', 'userLike'], axis=1, inplace=True)
    # print(df.info())
    return df


def getRatingData():
    ratings = list(db.collection(u'rating').stream())

    ratings_dict = list(map(lambda x: x.to_dict(), ratings))
    df = pd.DataFrame(ratings_dict, columns=['userId', 'hotelId', 'ratingPoint'])
    df['userId'] = pd.to_numeric(df['userId'])
    df['hotelId'] = pd.to_numeric(df['hotelId'])
    df['ratingPoint'] = pd.to_numeric(df['ratingPoint'])

    return df
