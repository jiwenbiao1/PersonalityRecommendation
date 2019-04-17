'''
模型类
'''
from .load_data import Data
import numpy as np
from .preprocessing import user_ratings_normalize, item_ratings_scaler
import tensorflow as tf

class LFMModel(object):
    def __init__(self, factor_n=10):
        data = Data()
        self.rating, self.record = data.get_rating_data()
        self.U_parameters = tf.Variable(tf.random.normal([factor_n, self.rating.shape[0]]))
        self.V_parameters = tf.Variable(tf.random.normal([factor_n, self.rating.shape[1]]))
        self.sess = None

    def _train(self, loss, learning_rate, steps, debug):
        self.sess = tf.Session()
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for i in range(steps):
            self.sess.run(train)
            if debug and i % 100 == 0:
                print('steps:', i + 1, 'loss:', self.sess.run(loss))

    def train(self, learning_rate=1e-2, steps=400, debug=False):
        loss = 1/2 * tf.reduce_sum( ( (tf.matmul(self.U_parameters, self.V_parameters, transpose_a = True) - self.rating)  ** 2) * self.record) + \
1/2 * 0.001 * ( tf.reduce_sum(self.U_parameters ** 2) + tf.reduce_sum(self.V_parameters ** 2) )
        self._train(loss, learning_rate, steps, debug)
        
    def score(self):
        Current_U_parameters, Current_V_parameters = self.sess.run([self.U_parameters, self.V_parameters])
        predicts = np.dot(Current_U_parameters.T, Current_V_parameters) 
        rmse, mae = self.model_metrics(predicts)
        print('RMSE:',rmse)
        print('MAE:', mae)
        return rmse, mae
    
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
        return rmse, mae





    
