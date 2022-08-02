import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dropout, Dense, GRU, LSTM, BatchNormalization,Activation,Bidirectional
from tensorflow.keras import Model,layers
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.framework import tensor_shape
import random
"""
                        tensorflow2.x , 必须搭配numpy1.18
涉及参数循环更新，使用tf.Variable方法设置初值，并用来保存更新的参数
        可以考虑把初值和更新放入网络循环中，再用get_Variable方法 
"""
####### 确定所需参数  ###########
params = {'momentum' :0.6,   #BN层指数加权平均的参数
          'units1':9,  'units2':2, #1和2分别是归一化与反归一化层的，用于设置w
          'shuchu':24, #输出的特征个数
          'a':-0.8,  'b':0.8,  #区间归一化的两个区间
          'epsilon':1e-5,  #根号计算时需要的微小正数
          }

data = pd.read_csv('室内温度预测30分钟采样.csv', encoding='ANSI').values
data = data[:8256,[1]]#8256(344*24)
data = np.array(data,dtype='float32').reshape(-1,24)
# x_train = data[:550,:]
# y_train = data[1:551,:]
# x_test = data[551:686,:]
# y_test = data[552:687,:]
x_train = data[:275,:].reshape(-1,24,1)
y_train = data[1:276,:]
x_test = data[276:343,:].reshape(-1,24,1)
y_test = data[277:344,:]


x_train,y_train= np.array(x_train,dtype='float32'),np.array(y_train,dtype='float32')
x_test,y_test = np.array(x_test,dtype='float32'),np.array(y_test,dtype='float32')
A,B = x_train.shape[1],x_train.shape[2]
units = x_train.shape[1]*x_train.shape[2] #18,把三维的转化为二维的，归一化完了之后，再转化为三维的
print(x_train.shape,x_train)  #(4941, 9, 2)  #(13912, 24, 6)
print(y_train.shape,y_train)  #(4941,)  #(13912, 24)
print(x_test.shape,x_test)  # (549, 9, 2)  #(3485, 24, 6)
print(y_test.shape,y_test)  # (549,)  #(3485, 24)

#这里用可变变量定义rm，rs的初值，并通过  assign() 方法对初值进行更新
#这里的初值必须设定的合适，参考数据的分布，太大的数据，必须能让他更新
running_mean = tf.Variable(initial_value=np.ones([units],
                                                  dtype='float32'), name='running_mean')
running_std = tf.Variable(initial_value=np.ones([units],
                                                 dtype='float32'), name='running_std')
running_max = tf.Variable(initial_value=np.ones([units],
                                                 dtype='float32'), name='running_max')
#设置的初始最小值是10，主要是根据数据集来设的，必须能让他更新
running_min = tf.Variable(initial_value=np.ones([units],
                                                 dtype='float32')*-10, name='running_min')
running_weiyi = tf.Variable(initial_value=np.ones([units],
                                                 dtype='float32')*30, name='running_weiyi')
########################四种归一化层##############################
class MyLayer(layers.Layer):
#并集合了四种归一化层，计算是基于二维的，可以将三维数据转化为二维数据
    def __init__(self, mode ='minmax'):
        super(MyLayer,self).__init__()
        print("Mode = ", mode)
        self.momentum = params['momentum']
        self.mode = mode
        self.weight = self.add_weight('weight',shape=[units, units])
        self.bias = self.add_weight('bias',shape=[units, ])

    def call(self, inputs,train_flag):
        #先计算并保存数据，然后可以用作返回值，有助于做反归一化
        inputs = tf.reshape(inputs,[-1,units])  #转化为二维的

        # 这里计算并保存最大最小值的更新数据：running_max, running_min
        datamin,datamax = tf.reduce_min(inputs),tf.reduce_max(inputs,axis=0)
        #datamin是一个数，但是为了更新和后续计算，把他复制多个
        running_ma = tf.add(tf.multiply(self.momentum, running_max),
                            tf.multiply(1 - self.momentum, datamax))
        running_mi = tf.add(tf.multiply(self.momentum, running_min),
                            tf.multiply(1 - self.momentum, datamin))
        running_max.assign(running_ma)
        running_min.assign(running_mi)

        datamean,variance = tf.nn.moments(inputs,axes=0)  #axes=0
        datastd = tf.sqrt(tf.add(variance,1e-5))

        # 这里计算并保存小数位移的数目weishu，需要一些格式转换
        absmax = tf.reduce_max(tf.abs(inputs))
        log10 = tf.divide(tf.math.log(absmax), tf.math.log(10.))
        weishu = tf.pow(10., tf.math.ceil(log10))
        weishu = tf.reshape(weishu,[1,])
        weishu = tf.tile(weishu,[units])
        # weishu = tf.maximum(weishu, running_weiyi)
        weishu = tf.add(tf.multiply(self.momentum, running_weiyi),
                            tf.multiply(1 - self.momentum, weishu))
        running_weiyi.assign(weishu)

        # (x-min) / (max-min)最小-最大规范化
        if self.mode == 'minmax':
            output = tf.divide(tf.subtract(inputs, running_min),
                               tf.subtract(running_max, running_min))
        #a+[(b-a)*(x-min)]/(max-min)区间标准化
        elif self.mode == 'qvjian':
            minmax = tf.divide( tf.subtract(inputs,running_min) ,
                                tf.subtract(running_max,running_min))
            qvjian = tf.multiply(params['b']-params['a'],minmax)
            output = tf.add(params['a'],qvjian)
        #小数定标规范化
        elif self.mode == 'weiyi':
            output = tf.divide(inputs, running_weiyi)
        # 批标准化
        elif self.mode == 'ozero' :
            if train_flag:
                output = tf.divide(tf.subtract(inputs, running_mean), tf.sqrt(tf.add(running_std,1e-5)))
                running_me = tf.add(tf.multiply(self.momentum, running_mean),
                                      tf.multiply(1 - self.momentum, datamean))
                running_st = tf.add(tf.multiply(self.momentum, running_std),
                                     tf.multiply(1 - self.momentum, variance))
                # 当前的running_mean是可以更新的，保存到循环外面的可变值里面
                running_mean.assign(running_me)
                running_std.assign(running_st)
            else:
                output = tf.divide(tf.subtract(inputs, running_mean), tf.sqrt(tf.add(running_std,1e-5)))
        else:
            return 0

        outputs = tf.matmul(output, self.weight) + self.bias
        outputs = tf.reshape(outputs,[-1,A,B])
        return outputs

class MyLayer2(layers.Layer):
#用于实现反归一化，承接的输入是二维的,但是需要接入两个函数的输入
#一个是刚刚输入到归一化层的x(inputs1)，另一个是GRU层的输出(inputs2)
#这里必须考虑GRU输出如何与关键值相乘，以及与权重w相乘，最终得到想要的输出格式
#现在的想法是：把GRU输出限制为(None,1)，方便和(1,18)相乘，权重设置为(18,1)
    def __init__(self,mode ='minmax'):
        super(MyLayer2,self).__init__()
        print("反归一化Mode = ", mode)
        self.mode = mode
        self.momentum = params['momentum']
        self.weight = self.add_weight('weight',shape=[units,params['shuchu']])
        self.bias = self.add_weight('bias',shape=[params['shuchu'], ])

    def call(self, inputs1,inputs2,train_flag):
        inputs1 = tf.reshape(inputs1, [-1, units])
        datamean, variance = tf.nn.moments(inputs1,axes=0)  #axes=0
        datastd = tf.sqrt(tf.add(variance,1e-5))

        # x*(max-min)+min 最大最小值反归一化
        if self.mode == 'minmax':
            output = tf.add( tf.multiply(inputs2,tf.subtract(running_max,running_min)) ,
                                running_min)
        #区间反归一化
        elif self.mode == 'qvjian':
            A = tf.divide(tf.subtract(inputs2,params['a']) , params['b']-params['a'])
            output = tf.add(tf.multiply(A, tf.subtract(running_max, running_min)),
                            running_min)
        #小数定标反归一化
        elif self.mode == 'weiyi':
            output = tf.multiply(inputs2,running_weiyi)
        #z-score反归一化
        elif self.mode == 'ozero':
            if train_flag:
                output = tf.add(tf.multiply(inputs2, tf.sqrt(tf.add(running_std,1e-5))), running_mean)
            else:
                output = tf.add(tf.multiply(inputs2, tf.sqrt(tf.add(running_std,1e-5))), running_mean)
        else:
            return 0
        return tf.matmul(output, self.weight) + self.bias

class MyModel(Model):
    def __init__(self,mode='minmax', train_flag=True):
        super(MyModel, self).__init__()
        self.mode = mode
        self.norm = MyLayer(mode=mode)
        self.act1 = Activation('tanh')  # 激活层1
        self.gru1 = GRU(96, input_shape=(A,B),return_sequences=True,activation='tanh')
        # self.gru1 = Bidirectional(LSTM(96, input_shape=(params['units1'], params['units2']), return_sequences=True, activation='tanh'))
        self.dropout = Dropout(0.2)
        self.gru2 = GRU(48,return_sequences=True,activation='tanh')
        # self.gru2 = Bidirectional(LSTM(48, return_sequences=True, activation='tanh'))
        self.dropout2 = Dropout(0.2)
        self.gru3 = GRU(24,activation='tanh')
        # self.gru3 = Bidirectional(LSTM(24, activation='tanh'))
        self.dence = Dense(1, activation='tanh')  #####没有激活函数！！
        self.fannorm = MyLayer2(mode=mode)
        self.train_flag = train_flag  #用它来调节训练和测试阶段

    def call(self, x):
        x1 = self.norm(x,self.train_flag)
        # x1 = tf.reshape(x1,[-1,params['units1'],params['units2']]) #单变量预测单变量时的转化
        # x2 = self.act1(x1)
        x3 = self.gru1(x1)
        x4 = self.dropout(x3)
        x5 = self.gru2(x4)
        x6 = self.dropout2(x5)
        x7 = self.gru3(x6)
        x8 = self.dence(x7)
        y = self.fannorm(x,x8,self.train_flag)
        return y


#方法选择模块及运行
predicted = []
evaluated = []
mode = ['weiyi']  #['ozero','minmax','qvjian','weiyi']
for mode in mode:
    print('mode是多少：',mode)
    model = MyModel(mode=mode)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss='mean_squared_error')
    history = model.fit(x=x_train, y=y_train, batch_size=30, epochs=500, validation_data=(x_test, y_test))
    # #绘制loss图像
    # valloss = history.history['val_loss']
    # loss = history.history['loss']
    # plt.plot(loss, label='loss')
    # plt.plot(valloss, label='valloss')
    # plt.legend()
    # plt.show()
    model.train_flag = False
    predict = model(x_test)
    pred = predict.numpy().reshape(-1, 1)
    real = y_test.reshape(-1, 1)

    RMSE = math.sqrt(mean_squared_error(real, pred))
    MAE=(mean_absolute_error(real, pred))
    MSE = (mean_squared_error(real, pred))
    MAPE = np.mean(np.abs((real - pred) / (real+1e-5)) * 100)
    real = np.squeeze(real)
    pred = np.squeeze(pred)
    R = (pearsonr(real, pred)[0])
    predicted.append(pred)

    evaluate = [mode,RMSE,MAE,MSE,MAPE,R]
    evaluated.append(evaluate)

real = y_test.reshape(-1,1)
#保存四种方法的预测值
predicted = np.array(predicted).T #这里集合了四列，mode

#画图，自己设定画第几列的，即mode
# plt.plot(predicted[1,:],label = '预测值')
# plt.plot(real,label='真实值')
# plt.legend()
# plt.show()

predict = pd.DataFrame(predicted)
real = pd.DataFrame(real)
# predict.to_csv('data_predict.csv', index=False,header=['ozero','minmax','qvjian','weiyi'])
# real.to_csv('data_real.csv', index=False)
# 选出最优的评价指标，找到多维矩阵中某列最小的值所在行数据
evaluated = np.array(evaluated)
print(evaluated[:,1])
hang = np.argmin(evaluated[:,1])
print('全部的结果是：RMSE,MAE,MSE,MAPE,R',evaluated)
print('最优的结果是：RMSE,MAE,MSE,MAPE,R',evaluated[hang,:])








