'''
模型类
'''
from .load_data import Data
import numpy as np
from .preprocessing import user_ratings_normalize, item_ratings_scaler
import tensorflow as tf

class LFMModel(object):
    def __init__(self, factor_n=10, random_state=666):
        data = Data(random_state=666)
        self.rating_train, self.rating_test = data.get_rating_data()
        self.U_parameters = tf.Variable(tf.random.normal([factor_n, self.rating_train.shape[0]]))
        self.V_parameters = tf.Variable(tf.random.normal([factor_n, self.rating_train.shape[1]]))
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

    def train(self, learning_rate=1e-2, steps=500, debug=False):
        record = (self.rating_train > 0).astype(int)
        loss = 1/2 * tf.reduce_sum( ( (tf.matmul(self.U_parameters, self.V_parameters, transpose_a = True) - self.rating_train)  ** 2) * record) + \
1/2 * 0.001 * ( tf.reduce_sum(self.U_parameters ** 2) + tf.reduce_sum(self.V_parameters ** 2) )
        self._train(loss, learning_rate, steps, debug)
        
    
    # 计算模型在测试集上的 RMSE 与 MAE
    def model_metrics(self, predict):
        record_index = self.rating_test > 0
        test_size = np.sum(record_index)
        
        matrix = (predict - self.rating_test)**2
        matrix[~record_index] = 0
        rmse = np.sqrt(np.sum(matrix) / test_size)
        
        matrix = np.abs((predict - self.rating_test))
        matrix[~record_index] = 0
        mae = np.sum(matrix) / test_size

        print("RMSE: ", rmse)
        print("MAE: ", mae)
    
        return rmse, mae

    def score(self):
        Current_U_parameters, Current_V_parameters = self.sess.run([self.U_parameters, self.V_parameters])
        predicts = np.dot(Current_U_parameters.T, Current_V_parameters) 
        rmse, mae = self.model_metrics(predicts)
        return rmse, mae
