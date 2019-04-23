import numpy as np
import pandas as pd
from .load_data import Data

# 对矩阵进行 item 归一化
def item_ratings_scaler(rating, record):
    # m代表用户数量，n代表服务数量
    m, n =rating.shape
    # 保存每个服务的平均响应时间
    rating_mean = np.zeros((n, ))
    # 保存经过正则化后的矩阵
    rating_norm = np.zeros((m,n))
    # 求每个服务的平均值，对每一列求均值
    for i in range(n):
        #第i个服务 对应用户评过分idx 平均得分；
        idx = record[:,i] != 0
        if not np.all(idx == 0):
                rating_mean[i] = np.mean(rating[idx, i])
                rating_norm[idx, i] = rating[idx, i] - rating_mean[i]
        
    return rating_norm, rating_mean

# 对矩阵进行 user 归一化
def user_ratings_normalize(rating, record):
    # m代表用户数量，n代表服务数量
    m, n =rating.shape
    # 保存每个用户的平均响应时间
    rating_mean = np.zeros((m, 1))
    # 保存经过正则化后的矩阵
    rating_norm = np.zeros((m,n))
    # 求每个用户的平均值，对每一列求均值
    for i in range(m):
        # 第i个用户 对应用户评过分idx 平均得分；
        idx = record[i,:] !=0
        if not np.all(idx == 0):
            rating_mean[i] = np.mean(rating[i, idx])
            rating_norm[i, idx] = rating[i, idx] - rating_mean[i]
        
    return rating_norm, rating_mean

# 解析评分行为的数据为BiNE格式
def parse_user_action(train_file='../../BiNE/data/rating_train.dat', test_file='../../BiNE/data/rating_test.dat'):
    # 获取训练使用的评分矩阵
    data = Data()
    rating = data.get_rating_data()[0]
    # record = rating > 0
    rows,columns = rating.shape
    train_list = []
    for row in range(rows):
        for column in range(columns):
            weight = rating[row, column]
            if weight != 0:
                train_list.append(['u' + str(row), 'i' + str(column), str(weight)])
    
    columns_list = ['source','target', 'weight']

    source_df = pd.DataFrame(train_list, columns=columns_list)

    # 乱序
    values = source_df.values
    np.random.shuffle(values)
    shuffle_df = pd.DataFrame(values, columns=columns_list)

    # 划分测试数据集
    ratio = int(source_df['source'].count() * 0.8)
    test_df = shuffle_df.iloc[ratio:, :]
    train_df = source_df

    # 保存数据
    train_df.to_csv(train_file, index=None, header=None, sep='\t')
    test_df.to_csv(test_file, index=None, header=None, sep='\t')


def _generate_BiNE_data(df):
    new_df = pd.DataFrame()
    # 转换 source
    new_df['source']  = df['source'].map(lambda x: 'u'+ str(x))
    # 转换 target
    data_set = set(df['target'])
    data_dict = {v:'i' + str(i) for i, v in enumerate(data_set)}
    new_df['target'] = df['target'].map(data_dict)
    new_df['weight'] = 1
    return new_df

# 解析电影的分类数据为BiNE格式
def parse_item_tags(train_file='../../BiNE/data/rating_train.dat', test_file='../../BiNE/data/rating_test.dat'):
    # 获得每部电影对应的分类二部图
    movie_df = pd.read_csv('data/ml-latest-small/movies.csv')
    movie_df['movie_row'] = movie_df.index
    tags_result = []
    for item in movie_df.itertuples():
        item_row = item.movie_row
        for tag in item.genres.split('|'):
            tags_result.append([item_row, tag])
    
    train_df = pd.DataFrame(tags_result, columns=['source','target'])

    # 乱序
    values = train_df.values
    np.random.shuffle(values)
    shuffle_df = pd.DataFrame(values, columns=['source','target'])

    # 划分测试数据集
    ratio = int(train_df['source'].count() * 0.8)
    test_df = shuffle_df.iloc[ratio:, :]

    # 处理成 BiNE 格式
    train_df = _generate_BiNE_data(train_df)
    test_df = _generate_BiNE_data(test_df)

    # 保存数据
    train_df.to_csv(train_file, index=None, header=None, sep='\t')
    test_df.to_csv(test_file, index=None, header=None, sep='\t')