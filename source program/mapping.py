import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras import Model,layers
tf.executing_eagerly()
matplotlib.rcParams['font.family'] = 'SimHei'   #这两句都是为了正确显示中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] =False#显示负号

#########################选择模块示意###############################

# data = [['mybn', 0.4002352715533772, 0.3185728, 0.16018827, 2.0113611, 0.9980831397636996],
#         ['minmax', 0.5952905195288389, 0.5213638, 0.3543708, 3.2820444, 0.998573518906165],
#         ['qvjian', 1.375083920778863, 1.331982, 1.8908558, 7.8308625, 0.9987586566240654],
#         ['weiyi', 0.8414345158298292, 0.76624626, 0.70801204, 4.407125, 0.9985354009488818]]
# data = np.array(data)
# hang = np.argmax(data[:,1])
# print('最优的结果是：',data[hang,:])

####################绘制归一化对比图像###########################
data = pd.read_csv('室内外温度数据1.csv')
data = data.iloc[2800:3300,[1]]
data = np.array(data)
# print(data)
a,b = -0.8,0.8
datamin,datamax,datamean,datavar = np.min(data), np.max(data), np.mean(data), np.var(data)

minmax = (data-datamin)/(datamax-datamin)
qvjian = a+(b-a)*(data-datamin)/(datamax-datamin)
j  = np.ceil(np.log10(np.max(abs(data))))
xiaoshu = data/(10**j)

print(xiaoshu)
ojunzhi = (data-datamean)/np.sqrt(datavar+1e-5)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,}
# plt.rcParams['figure.dpi'] = 300# 分辨率
plt.subplot(3,1,1)
plt.tick_params(width=2, labelsize=20)
plt.title('①',x=0.02,y=0.83,color='#EE2C2C',fontdict={'size':25 })
plt.ylabel('T \℃',font2)
plt.plot(data,color='#008BC5',linewidth=4.0)

plt.subplot(3,2,3)
plt.tick_params(width=2, labelsize=20)
plt.title('②',x=0.02,y=0.83,color='#EE2C2C',fontdict={'size':25 })
plt.plot(minmax,color='#008BC5',linewidth=4.0)

plt.subplot(3,2,4)
plt.tick_params(width=2, labelsize=20)
plt.title('③',x=0.02,y=0.83,color='#EE2C2C',fontdict={'size':25 })
plt.plot(qvjian,color='#008BC5',linewidth=4.0)

plt.subplot(3,2,5)
plt.tick_params(width=2, labelsize=20)
plt.title('④',x=0.02,y=0.83,color='#EE2C2C',fontdict={'size':25 })
plt.plot(xiaoshu,color='#008BC5',linewidth=4.0)

plt.subplot(3,2,6)
plt.tick_params(width=2, labelsize=20)
plt.title('⑤',x=0.02,y=0.83,color='#EE2C2C',fontdict={'size':25 })
plt.plot(ojunzhi,color='#008BC5',linewidth=4.0)
plt.show()

# #################  温湿度，光照，CO2数据画图展示   ######################
# data = pd.read_csv('室内温度预测30分钟采样.csv')
# data = data.iloc[:500,[1,2,3,5]]
# data = np.array(data)
# print(data)
# wendu = data[:,0]
# shidu = data[:,1]
# gq = data[:,2]
# co2 = data[:,3]
# # plt.title('温室大棚室内温湿度')
# # plt.rcParams['figure.dpi'] = 300# 分辨率
# font2 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 20,}
#
# plt.subplot(2,2,1)
# plt.tick_params(width=2, labelsize=15)
# plt.plot(wendu,color='#50ABA0',linewidth=4.0)
# plt.ylabel('Temperature (℃)',font2)
# plt.title('①',x=0.05,y=0.85,color='#EE2C2C',fontdict={'size':28})
#
# plt.subplot(2,2,2)
# plt.tick_params(width=2, labelsize=15)
# plt.plot(shidu,color='#50ABA0',linewidth=4.0)
# plt.ylabel('Humidity  (%RH)',font2)
# plt.title('②',x=0.05,y=0.85,color='#EE2C2C',fontdict={'size':28})
#
# plt.subplot(2,2,3)
# plt.tick_params(width=2, labelsize=15)
# plt.plot(co2,color='#50ABA0',linewidth=4.0)
# plt.ylabel('CO₂ (ppm)',font2)
# plt.title('③',x=0.05,y=0.85,color='#EE2C2C',fontdict={'size':28})
#
# plt.subplot(2,2,4)
# plt.tick_params(width=2, labelsize=15)
# plt.plot(gq,color='#50ABA0',linewidth=4.0)
# plt.ylabel('Light intensity  (Lux)',font2)
# plt.title('④',x=0.05,y=0.85,color='#EE2C2C',fontdict={'size':28})
#
# plt.show()
