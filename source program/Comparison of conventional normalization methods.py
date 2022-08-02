import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, GRU, BatchNormalization,LSTM,RNN,Embedding,Bidirectional,Flatten
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import os
from tensorflow.keras import Model,layers
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import math
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

data = pd.read_csv('室内温度预测30分钟采样.csv')
data = data.iloc[:8256,[1]] #8256
data = np.array(data)
data = data.reshape(-1,24)
print(data.shape)
x_train = data[:275,:].reshape(-1,24,1)/100
y_train = data[1:276,:]/100
x_test = data[276:343:,].reshape(-1,24,1)/100
y_test = data[277:344:,]/100
print('x_train:',x_train)
print('y_train',y_train)
print('x_test',x_test)
print('y_tets',y_test)

class Attention(layers.Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = tf.keras.initializers.get('glorot_uniform')
        # W_regularizer: 权重上的正则化
        # b_regularizer: 偏置项的正则化
        self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.b_regularizer = tf.keras.regularizers.get(b_regularizer)
        # W_constraint: 权重上的约束项
        # b_constraint: 偏置上的约束项
        self.W_constraint = tf.keras.constraints.get(W_constraint)
        self.b_constraint = tf.keras.constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)  #keras.backend.cast(x, dtype): 将张量转换到不同的 dtype 并返回

        if mask is not None:
            a *= K.cast(mask, K.floatx())#keras.backend.epsilon(): 返回浮点数

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

model = tf.keras.Sequential([
    #############attention-LSTM
    # LSTM(100, return_sequences=True),
    # Dropout(0.2),
    # LSTM(48,return_sequences=True),
    # Dropout(0.2),
    # LSTM(24,return_sequences=True),
    # Attention(24),
    # Dense(24)
    #############attention-GRU
    # GRU(100, return_sequences=True),
    # Dropout(0.2),
    # GRU(48,return_sequences=True),
    # Dropout(0.2),
    # GRU(24,return_sequences=True),
    # Attention(24),
    # Dense(24)
    ########dnese
    # Dense(100),
    # Dense(48),
    # Dense(12)
    #############GRU
    # GRU(100, return_sequences=True),
    # Dropout(0.2),
    # GRU(48,return_sequences=True),
    # Dropout(0.2),
    # GRU(24,return_sequences=True),
    # Dense(24)
    #############LSTM
    # LSTM(100, return_sequences=True),
    # Dropout(0.2),
    # LSTM(48,return_sequences=True),
    # Dropout(0.2),
    # LSTM(24,return_sequences=True),
    # Dense(24)
    ############BiGRU
    Bidirectional(GRU(100, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(GRU(100, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(GRU(100)),
    Dense(24)
    ################BiLSTM
    # Bidirectional(LSTM(100, return_sequences=True)),
    # Dropout(0.2),
    # Bidirectional(LSTM(100, return_sequences=True)),
    # Dropout(0.2),
    # Bidirectional(LSTM(100)),
    # Dense(24)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')  # 损失函数用均方误差
# 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值

history = model.fit(x_train, y_train, batch_size=24, epochs=500, validation_data=(x_test, y_test), validation_freq=1)
model.summary()

loss = history.history['loss']
val_loss = history.history['val_loss']

# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend()
# plt.show()

################## predict ######################
# 测试集输入模型进行预测
predict = model.predict(x_test)

# predict = ((predict+0.8)*(max3-min3))/1.6+min3
predict = predict*100
# print(predict)
# 对预测数据还原---从（0，1）反归一化到原始范围
# predict = sc.inverse_transform(predict)
predict = predict.reshape(-1,1)
# 对真实数据还原---从（0，1）反归一化到原始范围
# real = sc.inverse_transform(y_test.reshape(-1,3))
real = (y_test*100).reshape(-1,1)
print('real:',real)
print('predict',predict)

# real = y_test.reshape(-1,1)
# real = ((real+0.8)*(max4-min4))/1.6+min4

# real = sc.inverse_transform(y_test)
# 画出真实数据和预测数据的对比曲线
plt.plot(real, color='red', label='real')
plt.plot(predict, color='blue', label='predict')
plt.xlabel('epochs')
plt.legend()
plt.show()

##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
MSE = mean_squared_error(predict,real)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
RMSE = math.sqrt(mean_squared_error(predict,real))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
MAE = mean_absolute_error( predict,real)
MAPE = np.mean(np.abs((real,predict) / (real+1e-5))) * 100
real = np.squeeze(real)
predicted = np.squeeze(predict)
R=(pearsonr(predicted,real)[0])
print('RMSE:',RMSE,'MAE:',MAE,'MSE:',MSE,'MAPE:',MAPE,'R2:',R)

# real = pd.DataFrame(real)
predict = pd.DataFrame(predict)
# predict.to_csv('real.csv',index = False)
predict.to_csv('dense.csv',index = False)

'''
    BP1 = np.array([140.6614,120.5418,31.71])
    LSTM1 = np.array([129.4581,101.5268,48.98])
    GRU1 = np.array([128.6524,96.5924,48.81])
    # BiLSTM1 = np.array([77.6521,54.9854,15.5896,73.16])
    BiGRU1 = np.array([112.3584,85.6521,88.14])
    # LSTMAttention1 = np.array([69.7441,49.1254,10.9651,83.68])
    GRUAttention1 = np.array([100.7216,71.6594,98.27])
    AutoNorm1 = np.array([91.6691,66.6154,116.87])
'''