from flask import request
from flask import jsonify
from flask import Flask
import HotelRecommenderSys as recommend

app = Flask(__name__)


@app.route('/')
def hello():
    return jsonify('Welcome to BabyCare_RecommendSys server')



@app.route("/similar_hotel")
def similar_hotel():
    # Get hotel_id
    hotel_id = 0
    try:
        hotel_id = str(request.args.get('hotel_id'))
    except IndexError:
        hotel_id = 0

    # Init Recommender
    hotelRecommender = recommend.HotelRecommenderSys()

    # Return list hotelId
    hotelList = hotelRecommender.get_content_based(hotel_id)

    print(hotelList)
    return jsonify(hotelList)


@app.route("/hotel_recommend")
def hotel_recommend():
    # Get user_id
    user_id = 0
    try:
        user_id = int(request.args.get('user_id'))
    except IndexError:
        user_id = 0

    print(type(user_id))
    # Get recommend method
    nbcf = "0"
    try:
        nbcf = str(request.args.get('nbcf'))
    except IndexError:
        nbcf = "0"

    # Init Recommender
    hotelRecommender = recommend.HotelRecommenderSys()

    # Return list hotelId
    if nbcf == "0":
        hotelList = hotelRecommender.get_weighted_rating()
    else:
        hotelList = hotelRecommender.get_collaborative(user_id)

    print(hotelRecommender.get_weighted_rating())
    return jsonify(hotelList)


if __name__ == '__main__':
    app.run(host='localhost', port=3000, debug=True)
