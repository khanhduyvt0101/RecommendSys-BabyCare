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

C = products_data['ratePoint'].mean()  # C la diem trung binh cong rating cua tat ca cac product
m = products_data['rateCount'].quantile(0.75)  # m la so luong rating toi thieu cho mot san pham.
                                                #Day la ham tinh tu phan vi, voi diem phan vi la 0.75.
                                                # Vi the thi so luong Rating phai lon hon diem phan vi de thuat toan co the hoat dong.


# Lọc ra tất cả các sản phẩm đủ điều kiện (co du so luong rating) vào một DataFrame mới
q_products = products_data.copy().loc[products_data['rateCount'] >= m]


# ================================ #

# Hàm tính toán xếp hạng trọng số của từng sản phẩm
def weighted_rating(row, m=m, C=C):
    v = int(row['rateCount'])
    R = int(row['ratePoint'])
    return (v / (v + m) * R) + (m / (m + v) * C)


# Merge column values into one string and remove unuseful characters
def merge_info(row):
    baseString = str(row['name'] + ' ' + row['shopLocation'] + ' ' + row['tagName'] + '  ' + row['type']) # chuoi String dung de so sanh cac san pham tuong tu
    # Ham dung de xoa cac tu khong can thiet
    # unusefulWords = ['·', 'Vietnam', 'Việt Nam', ',', 'Phòng ', 'phòng ', '-']
    unusefulWords = []

    big_regex = re.compile('|'.join(map(re.escape, unusefulWords)))
    usefulString = big_regex.sub('', baseString)
    # Kết hợp số và từ thành một từ
    result = re.sub('(?<=\d) (?=\w)', '', usefulString)
    return result


# ========== Constant ============ #



class ProductRecommenderSys(object):
    def __init__(self):
        self.dfProducts = products_data
        self.dfRatings = ratings_data

    # Nhận đề xuất các sản phẩm dựa trên WEIGHTED RATING
    def get_weighted_rating(self):
        # Tao field moi cho q_products la field score và tính giá trị của nó với `weighted_rating () '
        q_products['score'] = q_products.apply(weighted_rating, axis=1)

        # Sắp xếp sản phẩm dựa trên số điểm đã tính ở trên
        result = q_products.sort_values('score', ascending=False)

        # Lay top 6 san pham dạt điểm cao nhất
        return result['id'][:7].to_list()

    # Lay đề xuất sản phẩm dựa trên Content-based Filtering
    def get_content_based(self, id):
        df = self.dfProducts
        # Lọc ra tất cả các sản phẩm đủ điều kiện dựa trên tagName vào một DataFrame mới
        row_index_matchId = df.index[df['id'] == id].tolist()
        df = df.copy().loc[df['tagName'] == df['tagName'].values[row_index_matchId[0]]]

        # Xây dựng map dao ngược của các chỉ số và idProduct
        indices = pd.Series(df.index, index=df['id']).drop_duplicates()

        # Lay vi tri của sản phẩm trong map o tren phù hợp với idProduct
        idx = indices[id]

        # Columns chua cac field can so sanh
        columns = ['name', 'shopLocation', 'tagName', 'type']

        # Chuyển đổi giá trị cột thành String idProduct
        for column in columns:
            df[column] = df[column].apply(str)

        # Tạo field moi la mergeInfo bằng cách áp dụng phương thức merge_info
        df['mergeInfo'] = df.apply(merge_info, axis=1)

        # Define bien tfidf la mot ma tran vector TF-IDF
        tfidf = TfidfVectorizer()
        # Xây dựng ma trận TF-IDF cần thiết bằng cách điều chỉnh và biến đổi dữ liệu
        tfidf_matrix = tfidf.fit_transform(df['mergeInfo'])

        # Tính toán ma trận bang phuong phap cosine_similarity
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Get cac cap co diem giong tuong tu voi san pham dang xet voi tong cac san pham cung tagName trong DB
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))

        # Sắp xếp các sản phẩm dựa trên điểm giống nhau
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Lay điểm của 6 sản phẩm giống nhất
        sim_scores = sim_scores[1:7]

        # Lay cac vi tri cua san pham
        products_indices = [i[0] for i in sim_scores]

        result = df['id'].iloc[products_indices]

        # Trở ve top 6 sản phẩm tương tự nhất
        return result.to_list()

