'''
数据载入模块
'''
import numpy as np
import pandas as pd

class Data():
    '''
    raing_file: 评分文件路径
    movie_file: 电影详情文件路径
    '''
    def __init__(self, rating_file="data/ml-latest-small/ratings.csv", movie_file="data/ml-latest-small/movies.csv" ):
        self.rating_file = rating_file
        self.movie_file = movie_file

    # 处理评分数据
    def processed_rating(self):
        rating_df = pd.read_csv(self.rating_file)
        movie_df = pd.read_csv(self.movie_file)
        movie_df['movie_row'] = movie_df.index

        rating_processed_df = rating_df.merge(movie_df, on='movieId')
        rating_processed_df['user_row'] = rating_processed_df['userId'].apply(lambda x: x - 1 )

        columns = ['movie_row', 'user_row', 'rating', 'title', 'genres']
        rating_processed_df = rating_processed_df[columns]

        rating_processed_df.to_csv('data/ml-latest-small/ratings_processed.csv', index=None)

        return rating_processed_df

    # 产生评分矩阵
    def genarate_rating_matrix(self):
        try:
            rating_df = pd.read_csv('data/ml-latest-small/ratings_processed.csv')
        except Exception:
            rating_df = self.processed_rating()
        
        user_n = rating_df.user_row.max() + 1
        item_n = rating_df.movie_row.max() + 1
        rating = np.zeros((user_n, item_n))
        for item in rating_df.itertuples():
            rating[item.user_row, item.movie_row] = item.rating
        df = pd.DataFrame(rating)
        df.to_csv('data/ml-latest-small/ratings_matrix.csv', header=None, index=None)
        return df

    # 获取评分矩阵 和 评分记录
    def get_rating_data(self):
        try:
            rating_df = pd.read_csv('data/ml-latest-small/ratings_matrix.csv', header=None)
        except Exception:
            rating_df = self.processed_rating()
        rating = rating_df.values
        return rating, (rating > 0).astype(int)