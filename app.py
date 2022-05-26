from flask import request
from flask import jsonify
from flask import Flask
import ProductRecommenderSys as recommend

app = Flask(__name__)


@app.route('/')
def hello():
    return jsonify('Welcome to BabyCare_RecommendSys server')



@app.route("/similar_products")
def similar_product():
    # Get product_id
    product_id = 0
    try:
        product_id = str(request.args.get('product_id'))
    except IndexError:
        product_id = 0

    # Init Recommender
    productRecommender = recommend.ProductRecommenderSys()

    # Return list productId
    productList = productRecommender.get_content_based(product_id)

    print(productList)
    return jsonify(productList)


@app.route("/product_recommend")
def product_recommend():
    # Get user_id
    # user_id = 0
    # try:
    #     user_id = int(request.args.get('user_id'))
    # except IndexError:
    #     user_id = 0

    # print(type(user_id))
    # # Get recommend method
    nbcf = "0"
    # try:
    #     nbcf = str(request.args.get('nbcf'))
    # except IndexError:
    #     nbcf = "0"

    # Init Recommender
    productRecommender = recommend.ProductRecommenderSys()

    # Return list hotelId
    if nbcf == "0":
        productList = productRecommender.get_weighted_rating()
    else:
        productList = productRecommender.get_collaborative(user_id)

    print(productRecommender.get_weighted_rating())
    return jsonify(productList)


if __name__ == '__main__':
    app.run(host='localhost', port=3000, debug=True)
