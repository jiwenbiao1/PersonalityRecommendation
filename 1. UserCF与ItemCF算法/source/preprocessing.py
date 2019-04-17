import numpy as np
import pandas as pd

# 解析评分行为的数据为BiNE格式
def parse_user_action(rating, record, train_file='../BiNE/data/rating_train.dat', test_file='../BiNE/data/rating_test.dat'):
    # 获取评分矩阵
    record = record.astype(bool)
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


# 解析电影的分类数据为BiNE格式
def parse_item_tags():
    pass