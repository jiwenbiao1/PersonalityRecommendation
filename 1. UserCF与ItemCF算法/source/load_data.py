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
    def __init__(self, random_state=666, rating_file="data/ml-latest-small/ratings.csv", movie_file="data/ml-latest-small/movies.csv" ):
        self.rating_file = rating_file
        self.movie_file = movie_file
        self.random_state = random_state
        self.rating_source = None # 处理之后的原始评分数据df

    # 处理评分数据
    def processed_rating(self):
        rating_df = pd.read_csv(self.rating_file)
        movie_df = pd.read_csv(self.movie_file)
        movie_df['movie_row'] = movie_df.index

        rating_processed_df = rating_df.merge(movie_df, on='movieId')
        rating_processed_df['user_row'] = rating_processed_df['userId'].apply(lambda x: x - 1 )

        columns = ['movie_row', 'user_row', 'rating']
        rating_processed_df = rating_processed_df[columns]

        rating_processed_df.to_csv('data/ml-latest-small/ratings_processed.csv', index=None)

        return rating_processed_df

    
    # 划分测试集与数据集
    def split_data(self):
        try:
            source_df = pd.read_csv('data/ml-latest-small/ratings_processed.csv')
        except Exception:
            source_df = self.processed_rating()
        
        self.rating_source = source_df

        user_max = source_df.user_row.max()
        
        for user_row in range(user_max + 1):
            source = source_df[source_df.user_row == user_row].values
            # 乱序
            np.random.seed(self.random_state)
            np.random.shuffle(source)
            size = source.shape[0]
            ratio = int(0.8 * size)
            train = source[:ratio, :]
            test = source[ratio:, :]
            if user_row == 0:
                train_result = train
                test_result = test
            else:
                train_result = np.vstack((train_result, train))
                test_result = np.vstack((test_result, test))
            
        rating_train = pd.DataFrame(train_result, columns=source_df.columns)
        rating_test = pd.DataFrame(test_result, columns=source_df.columns)

        return rating_train, rating_test
    
    # 产生评分矩阵
    def genarate_rating_matrix(self, rating_df):
        user_n = self.rating_source.user_row.max() + 1
        item_n = self.rating_source.movie_row.max() + 1
        rating = np.zeros((user_n, item_n))
        for item in rating_df.itertuples():
            rating[item.user_row, item.movie_row] = item.rating
        return rating

    # 产生评分的数据
    def get_rating_data(self):
        train_df, test_df = self.split_data()
        return self.genarate_rating_matrix(train_df), self.genarate_rating_matrix(test_df)

