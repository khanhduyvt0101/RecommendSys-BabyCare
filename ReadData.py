import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Use a service account
cred = credentials.Certificate('firebase_key/baby-care-69c67-93d5cb595f9f.json')
firebase_admin.initialize_app(cred)

db = firestore.client()


# Get Data from firebase
def getProductsData():
    products = list(db.collection(u'product').stream())

    products_dict = list(map(lambda x: x.to_dict(), products))
    df = pd.DataFrame(products_dict)

    # Drop unuseful feature
    df.drop(columns=['primaryImage', 'salePercent', 'basePrice', 'url', 'totalBought'], axis=1, inplace=True)
    # print(df.info())
    return df


def getRatingData():
    ratings = list(db.collection(u'Rating').stream())

    ratings_dict = list(map(lambda x: x.to_dict(), ratings))
    df = pd.DataFrame(ratings_dict, columns=['userId', 'idProduct', 'ratePoint'])
    # df['userId'] = pd.to_numeric(df['userId'])
    # df['idProduct'] = pd.to_numeric(df['idProduct'])
    df['ratePoint'] = pd.to_numeric(df['ratePoint'])

    return df
