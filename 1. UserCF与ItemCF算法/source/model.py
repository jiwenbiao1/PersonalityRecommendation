'''
模型类
'''
from .load_data import Data
import numpy as np
from .preprocessing import user_ratings_normalize, item_ratings_scaler

class CFModel(object):
    def __init__(self):
        data = Data()
        self.rating, self.record = data.get_rating_data()

        # 计算两个向量的 cos 距离
    def cosin_dis(self, x, y):
        div = (np.linalg.norm(x) * np.linalg.norm(y))
        return np.dot(x, y) / div

    # 为matrix[0] 生成相似度矩阵
    def generate_sim_matrix(self, matrix):
        self.matrix = matrix
        item_num = matrix.shape[0]
        sim_matrix = np.zeros(shape=(item_num, item_num))
        for row in range(item_num):
            for column in range(item_num):
                if row < column : 
                    sim =  self.get_pcc_similarity(row, column)
                    sim_matrix[row, column] = sim
                    sim_matrix[column, row] = sim        
        for row in range(item_num):
            row_sum = sim_matrix[row].sum()
            if row_sum != 0:
                sim_matrix[row] = sim_matrix[row] / sim_matrix[row].sum()
        return sim_matrix

    # matrix 为归一化后的评分矩阵,根据 PCC 求得矩阵 i, j 行的相似性 
    def get_pcc_similarity(self, i, j):
        # 分别获取用户i、j调用的服务索引
        i_index = self.matrix[i] > 0
        j_index = self.matrix[j] > 0
        # 获取用户i, j共同调用
        index = i_index & j_index
        # 如果 i j没有调用过相似的服务
        if index.sum() == 0:
            return 0
        # 计算PCC
        x = self.matrix[i,index]
        y = self.matrix[j, index]
        sim = self.cosin_dis(x, y)
        # 忽略pcc小于0的
        if sim < 0:
            sim = 0
        return sim
    
    # 计算模型的 RMSE 与 MAE
    def model_metrics(self, predict):
        record_index = self.record.astype(bool)
        test_size = np.sum(record_index)
        
        matrix = (predict - self.rating)**2
        matrix[~record_index] = 0
        rmse = np.sqrt(np.sum(matrix) / test_size)
        
        matrix = np.abs((predict - self.rating))
        matrix[~record_index] = 0
        mae = np.sum(matrix) / test_size

        print("RMSE: ", rmse)
        print("MAE: ", mae)
    
        return rmse, mae

# 基于用户的协同过滤
class UserCFModel(CFModel):
    def __init__(self):
        CFModel.__init__(self)
        rating_norm, rating_mean = user_ratings_normalize(self.rating, self.record) 
        self.sim_matrix = self.generate_sim_matrix(rating_norm)
        predict_matrix_T = np.dot(rating_norm.T, self.sim_matrix)
        self.predict_matrix = np.transpose(predict_matrix_T) + rating_mean
    
    def score(self):
        self.model_metrics(self.predict_matrix)

# 基于项目的协同过滤
class ItemCFModel(CFModel):
    def __init__(self):
        CFModel.__init__(self)
        rating_norm, rating_mean = item_ratings_scaler(self.rating, self.record) 
        self.sim_matrix = self.generate_sim_matrix(rating_norm.T)
        predict_matrix_T = np.dot(self.sim_matrix, rating_norm.T)
        self.predict_matrix = np.transpose(predict_matrix_T) + rating_mean
    
    def score(self):
        self.model_metrics(self.predict_matrix)
        # self.sim_matrix = self.generate_sim_matrix(self.rating_norm.T)

    
