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
另外，还存在对于多变量数据，进归一化网络时需要是二维的，这样就需要
在model的归一化层之后加入一个格式转化的层

"""
####### 确定所需参数  ###########
params = {'momentum' :0.8,   #BN层指数加权平均的参数
          'units1':24,  'units2':24, #1和2分别是归一化与反归一化层的，用于设置w
          'a':-0.8,  'b':0.8,  'weishu':100,#区间归一化的两个区间
          'epsilon':1e-5,  #根号计算时需要的微小正数
          'datamax':40. ,  'datamin': -20.,  #最大最小化时预先规定的最大最小参数
          }

data = pd.read_csv('室内温度预测30分钟采样.csv', encoding='ANSI').values
data = data[:8256,[1]]#8256(344*24)
data = np.array(data,dtype='float32').reshape(-1,24)
# x_train = data[:550,:]
# y_train = data[1:551,:]
# x_test = data[551:686,:]
# y_test = data[552:687,:]
x_train = data[:275,:]
y_train = data[1:276,:]
x_test = data[276:343,:]
y_test = data[277:344,:]
print("x_train.shape", x_test,x_test.shape)
print("y_train.shape", y_test,y_test.shape)

########################四种归一化层##############################
#这里用可变变量定义rm，rs的初值，并通过  assign() 方法对初值进行更新
# running_mean = tf.Variable(initial_value=np.zeros(params['units1'],
#                                                   dtype='float32'), name='running_mean')
# running_std = tf.Variable(initial_value=np.zeros(params['units1'],
#                                                  dtype='float32'), name='running_std')
running_max = tf.Variable(initial_value=np.zeros(params['units1'],
                                                 dtype='float32')*80, name='running_max')
running_min = tf.Variable(initial_value=np.ones(params['units1'],
                                                 dtype='float32'), name='running_min')
running_mean = tf.Variable(initial_value=np.zeros(params['units1'],
                                                  dtype='float32'), name='running_mean')
running_std = tf.Variable(initial_value=np.ones(params['units1'],
                                                 dtype='float32'), name='running_std')
running_weiyi = tf.Variable(initial_value=np.zeros(params['units1'],
                                                 dtype='float32'), name='running_weiyi')
class MyLayer(layers.Layer):
#定义并集合了所有的归一化层，需要输入是二维的
    def __init__(self, units,mode ='minmax'):
        super(MyLayer,self).__init__()
        print("Mode = ", mode)
        self.momentum = params['momentum']
        self.mode = mode
        self.units = units
        self.weight = self.add_weight('weight',shape=[self.units, self.units])
        self.bias = self.add_weight('bias',shape=[self.units, ])

    def call(self, inputs,train_flag):
        #先计算并保存数据，然后可以用作返回值，有助于做反归一化
        datamin,datamax = tf.reduce_min(inputs),tf.reduce_max(inputs)
        datamin = tf.tile(tf.reshape(datamin,[1,]),[params['units1']])
        datamax = tf.tile(tf.reshape(datamax, [1, ]), [params['units1']])
        print('datamax++', datamax)
        running_ma = tf.add(tf.multiply(self.momentum, running_max),
                            tf.multiply(1 - self.momentum, datamax))
        running_mi = tf.add(tf.multiply(self.momentum, running_min),
                            tf.multiply(1 - self.momentum, datamin))
        running_max.assign(running_ma)
        running_min.assign(running_mi)
        print('running_max++',running_max)

        datamean,variance = tf.nn.moments(inputs,axes=0)  #axes=0
        datastd = tf.sqrt(tf.add(variance,1e-5))
        absmax = tf.reduce_max(tf.abs(inputs))
        log10 = tf.divide(tf.math.log(absmax), tf.math.log(10.))
        weishu = tf.pow(10., tf.math.ceil(log10))
        weishu = tf.reshape(weishu,[1,])
        weishu = tf.tile(weishu,[params['units1']])
        weishu = tf.maximum(weishu, running_weiyi)
        running_weiyi.assign(weishu)
        # weishu = tf.add(tf.multiply(self.momentum, running_weiyi),
        #                     tf.multiply(1 - self.momentum, weishu))
        # running_weiyi.assign(weishu)
        # (x-min) / (max-min)最小-最大规范化
        if self.mode == 'minmax':
            # if train_flag:
            #     output = tf.divide(tf.subtract(inputs, datamin),
            #                    tf.subtract(datamax, datamin))
            # else:
            #     output = tf.divide(tf.subtract(inputs, running_min),
            #                    tf.subtract(running_max, running_min))
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
        elif self.mode == 'mybn' :
            if train_flag:
                output = tf.divide(tf.subtract(inputs, datamean), datastd)
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
        return tf.matmul(output, self.weight) + self.bias

class MyLayer2(layers.Layer):
#用于实现反归一化，承接的输入是一维的,但是需要接入两个函数的输入
#一个是刚刚输入到归一化层的x(inputs1)，另一个是GRU层的输出(inputs2)
    def __init__(self,units,mode ='minmax'):
        super(MyLayer2,self).__init__()
        print("反归一化Mode = ", mode)
        self.mode = mode
        self.units  = units
        self.momentum = params['momentum']
        self.weight = self.add_weight('weight',
            shape=[self.units, self.units])
        self.bias = self.add_weight('bias',
                shape=[self.units, ])

    def call(self, inputs1,inputs2,train_flag):
        datamin, datamax = tf.reduce_min(inputs1), tf.reduce_max(inputs1)
        datamean, variance = tf.nn.moments(inputs1,axes=0)  #axes=0
        datastd = tf.sqrt(tf.add(variance,1e-5))

        absmax = tf.reduce_max(tf.abs(inputs1))
        log10 = tf.divide(tf.math.log(absmax), tf.math.log(10.))
        weishu = tf.pow(10., tf.math.ceil(log10))
        # x*(max-min)+min 最大最小值反归一化
        if self.mode == 'minmax':
            # if train_flag:
            #     output = tf.add( tf.multiply(inputs2,tf.subtract(datamax,datamin)) ,
            #                     datamin)
            # else:
            #     output = tf.add( tf.multiply(inputs2,tf.subtract(running_max,running_min)) ,
            #                     running_min)
            output = tf.add( tf.multiply(inputs2,tf.subtract(running_max,running_min)) ,
                                running_min)
        #区间反归一化
        elif self.mode == 'qvjian':
            A = tf.divide(tf.subtract(inputs2,params['a']) , params['b']-params['a'])
            output = tf.add(tf.multiply(A, tf.subtract(running_max, running_min)),
                            running_min)
        #[-1,1]小数位移反归一化
        elif self.mode == 'weiyi':
            output = tf.multiply(inputs2,running_weiyi)
        #批标准化反归一化
        elif self.mode == 'mybn':
            if train_flag:
                output = tf.add(tf.multiply(inputs2, datastd), datamean)
            else:
                output = tf.add(tf.multiply(inputs2, tf.sqrt(tf.add(running_std,1e-5))), running_mean)
        else:
            return 0
        return tf.matmul(output, self.weight) + self.bias

class MyModel(Model):
    def __init__(self,mode='minmax', train_flag=True):
        super(MyModel, self).__init__()
        self.mode = mode
        self.norm = MyLayer(units=params['units1'],mode=mode)
        self.act1 = Activation('tanh')  # 激活层1
        self.gru1 = GRU(100,input_shape=(params['units1'],1),return_sequences=True,activation='tanh')
        self.dropout = Dropout(0.2)
        self.gru2 = GRU(48,return_sequences=True,activation='tanh')
        self.gru3 =GRU(24,return_sequences=True,activation='tanh')
        self.gru4 = GRU(24, activation='tanh')
        self.dence = Dense(24, activation='tanh')  #####没有激活函数！！
        self.fannorm = MyLayer2(units=params['units2'],mode=mode)
        self.train_flag = train_flag  #用它来调节训练和测试阶段

    def call(self, x):
        x1 = self.norm(x,self.train_flag)
        x2 = tf.reshape(x1,[-1,params['units1'],1]) #单变量预测单变量时的转化
        # x2 = self.act1(x2)
        x3 = self.gru1(x2)
        x4 = self.dropout(x3)
        x5 = self.gru2(x4)
        x6 = self.dropout(x5)
        x7 = self.gru3(x6)
        x7 = self.dropout(x7)
        x7 = self.gru4(x7)
        x8 = self.dence(x7)
        y = self.fannorm(x,x8,self.train_flag)
        return y

#归一化方法选择模块及运行
predicted = []
evaluated = []
mode = ['ozero','minmax','qvjian','weiyi']
for mode in mode:
    print('mode是多少：',mode)
    model = MyModel(mode=mode)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                      loss='mean_squared_error')
    history = model.fit(x=x_train, y=y_train, batch_size=60, epochs=10, validation_data=(x_test, y_test))
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
predicted = np.array(predicted).T  #这里集合了四列，mode

#画图，自己设定画第几列的，即mode
# plt.plot(predicted[1,:],label = '预测值')
# plt.plot(real,label='真实值')
# plt.legend()
# plt.show()

predict = pd.DataFrame(predicted)
real = pd.DataFrame(real)
predict.to_csv('data_predict.csv', index=False,header=['ozero','minmax','qvjian','weiyi'])
real.to_csv('data_real.csv', index=False)
# 选出最优的评价指标，找到多维矩阵中某列最小的值所在行数据
evaluated = np.array(evaluated)
hang = np.argmin(evaluated[:,1])
print('全部的结果是：RMSE,MAE,MSE,MAPE,R',evaluated)
print('最优的结果是：RMSE,MAE,MSE,MAPE,R',evaluated[hang,:])



