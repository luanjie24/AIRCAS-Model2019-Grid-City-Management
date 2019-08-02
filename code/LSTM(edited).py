# -*- coding: utf-8 -*-
#以下模块在本代码中并未使用到，在此留下以防万一
#from __future__ import print_function
#from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Input, Embedding, merge
from keras.utils import np_utils
from keras.models import Model
import pandas as pd
import numpy as np
from keras.optimizers import Adam
from keras.regularizers import l2
import csv
import random 
import os
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from matplotlib.font_manager import _rebuild
#from sklearn import preprocessing
#如需进行数据归一化则写入下一行代码
from sklearn.preprocessing import MinMaxScaler

#设置绘图时的中文显示（需安装黑体字体）
_rebuild()
mpl.rcParams['font.sans-serif']=[u'SimHei']
mpl.rcParams['axes.unicode_minus']=False
#读取文件
file_name = u"非法小广告-所有街道数据.csv"
df=pd.read_csv(file_name,encoding='gbk')

#获得表格第4列的所有数据，也就是立按量
hourlyData=df.values[:,3]
#定义数据长度datalen用于计算站点数量
datalen = df.shape[0]
#获得表格第3列第所有数据，也就是月份Mon
Mon=df.values[:,2]

#写成科学计数法（float32）
#hourlyData=hourlyData.astype('float32')

#输出一下得到的案件量个数
print('aqi data length:', len(hourlyData))

#读取数据
delay=10
num_Data=48
#站点数量=数据长度/48（其中48为4年的总月份数）
num_sites=int(datalen/num_Data)
X=[]
y=[]
pre=1

#立案量，转换成6行48列，每行代表不同的街道，每个街道有48个月的数据
hourlyData = hourlyData.reshape(num_sites,num_Data)

#转置 现在每一列代表不同的街道了
hourlyData = hourlyData.T

#====================================以下为对训练样本的处理 标准化、打乱等================================
#在进行运算之前可以对数据进行归一化，进而降低loss
scaler = MinMaxScaler(feature_range=(0.001, 1))
hourlyData = scaler.fit_transform(hourlyData)

for d in range(delay,len(hourlyData)-pre+1):
    X_one=hourlyData[d-delay:d,:]
    X_one=X_one.reshape((1,X_one.shape[0],X_one.shape[1]))
    y_one=hourlyData[d,:]
    X.append(X_one)
    y.append(y_one)
X=np.array(X).reshape((len(X),delay,num_sites))
y=np.array(y)

#shuffle data
random.seed(10)
random.shuffle(X)
random.seed(10)
random.shuffle(y)
random.seed(10)
random.shuffle(Mon)

#split dataset
trLen=int(0.8*X.shape[0])
train_set_x=X[:trLen,:]
train_set_y=y[:trLen]
test_set_x = X[trLen:,:]
test_set_y=y[trLen:]
#====================================================================

#==========================================本模块采用LSTM建模================================================
# build the model: 2 stacked LSTM
print('Build model...')
input_shape = (delay,num_sites)
main_input = Input(shape=input_shape, name='main_input')
rnn_out = LSTM(500, return_sequences=True,consume_less = 'gpu')(main_input)
x = LSTM(500,consume_less = 'gpu')(rnn_out)

#4、在后面连接一个隐层，输入为rnn输出和时间信息，采用sigmoid激活
x = Dense(500, activation='relu')(x)
#5、添加一个dropout层防止过拟合
x = Dropout(0.5)(x)
#6、后面添加一个隐层，采用relu作为激活函数，根据relu的特性，可以直接输出实数
#x = Dense(100, activation='relu')(x)
#7、继续使用relu输出最终预测值
loss = Dense(num_sites, activation='relu', name='main_output')(x)

#使用刚才创建的图生成模型
model = Model(input=[main_input], output=[loss])

solver = Adam(lr=0.001)
model.compile(optimizer=solver,
                  loss={'main_output': 'mape'} )

#=============================================================================================

#定义精度计算公式
def cal_acc(pre,real):
    pre = scaler.inverse_transform(pre)
    real = scaler.inverse_transform(real)
    [m,n]=pre.shape
    pre=pre.reshape(m*n,1)
    real=real.reshape(m*n,1)
    acc=np.zeros((4,1))
    acc[0]=np.sqrt(((pre-real)**2).mean())
    acc[1]=(abs(pre-real)).mean()
    acc[2]=(abs(pre-real)/real).mean()
    acc[3]=1-sum((pre-real)**2)/sum((abs(pre-real.mean())+abs(real-real.mean()))**2)
    return acc.transpose()
#生成.json和.hdf5文件用于预测
model_json = model.to_json()
model_path = '$8.json'
model_weight_path = '$8_weights.hdf5'
with open(model_path, "w") as json_file:
    json_file.write(model_json)
#定义总迭代次数epoches和初始精度数组acc_tr和acc_t
epoches = 100
acc_tr=np.zeros((epoches,4))
acc_t=np.zeros((epoches,4))
history = []
#================================================================训练LSTM=====================================
#开始迭代
for epoch in range(epoches):
    print()
    print('-' * 50)
    print('epoch', epoch)
    if epoch==50:
        solver = Adam(lr=0.0001)
        model.compile(optimizer=solver,
                  loss={'main_output': 'mape'} )
    hist = model.fit({'main_input': train_set_x},
                  {'main_output': train_set_y},validation_data=(
                    {'main_input': test_set_x, },
                    {'main_output': test_set_y}
                  ),verbose = 1,
                  nb_epoch=10, batch_size=128)
    acc_tr[epoch,:]=cal_acc(model.predict([train_set_x]),train_set_y)
    acc_t[epoch,:]=cal_acc(model.predict([test_set_x]),test_set_y)
    #msemae_tr[epoch,:] = cal_msemae_tr()
    #msemae_t[epoch,:] = cal_msemae_t()
    history.extend(hist.history.values())
history = np.array(history).reshape((-1,1))
if model_weight_path:
    if os.path.exists(model_weight_path):
        os.remove(model_weight_path)
    model.save_weights(model_weight_path) # eg: model_weight.h5
#========================================================================================================

#输出精度acc
a=[acc_tr[:,3],acc_t[:,3]]
a=np.array(a)
a=a.T

#定义预测值
trainPredict = model.predict(train_set_x)
testPredict = model.predict(test_set_x)

#反归一化：如在开始时进行了归一化则取消以下代码的注释
hourlyData = scaler.inverse_transform(hourlyData)
trainPredict = scaler.inverse_transform(trainPredict)
train_set_y = scaler.inverse_transform(train_set_y)
testPredict = scaler.inverse_transform(testPredict)
test_set_y = scaler.inverse_transform(test_set_y)

#将数据按站点分为SITE_SIZE组
site_names=[]  #站点数据列表
site_cnames=[] #站点名字列表
for num in range(0,num_sites):
    site_cnames.append(df.at[num*num_Data, u'事发街道'])
    site_names.append(hourlyData[:,num])

#作图：立案量vs时间
plt.figure(figsize=(16,9))
plt.suptitle(u'各地点按月立案数量')
layout_num = 0
for i in range(0,num_sites):
    if(layout_num==6):
        layout_num=0
        plt.figure(figsize=(16, 9))
        plt.suptitle(u'各地点按月立案数量')
    subplot = plt.subplot(3, 2, layout_num + 1)
    site = site_names[i]
    plt.plot(site_names[i])
    plt.xlabel(u'时间')
    plt.ylabel(u'立案量')
    plt.legend(labels=[u'立案量'],loc = 'best')
    subplot.set_title(site_cnames[i])
    plt.tight_layout()
    layout_num = layout_num + 1

#作图：精度vs迭代次数
plt.figure(figsize=(16,9))
#x2 = np.linspace(1,100,len(a[:,0]))
#x21 = np.linspace(1,100,len(a[:,1]))
plt.plot(a[:,0])
plt.plot(a[:,1])
plt.xlabel(u'迭代次数')
plt.ylabel(u'精度')
plt.title(u'测试/训练精度与时间对比')
plt.legend(labels = [u'训练精度',u'测试精度'],loc ='best')

#作图：真实值&预测值vs时间
#训练数据组

plt.figure(figsize=(16,9))
plt.suptitle(u'分站点预测/实际值对比（训练数据）')
layout_num = 0
for i in range(0,num_sites):
    if(layout_num==6):
        layout_num=0
        plt.figure(figsize=(16, 9))
        plt.suptitle(u'分站点预测/实际值对比（训练数据）')
    subplot = plt.subplot(3, 2, layout_num + 1)
    #site = site_names[i]
    plt.plot(trainPredict[:, i])
    plt.plot(train_set_y[:, i])
    plt.xlabel(u'数据编号')
    plt.ylabel(u'数值')
    plt.legend(labels=[u'预测数据',u'实际数据'])
    subplot.set_title(site_cnames[i])
    plt.tight_layout()
    layout_num = layout_num + 1

#测试数据组

plt.figure(figsize=(16,9))
plt.suptitle(u'分站点预测/实际值对比（测试数据）')
layout_num = 0
for i in range(0,num_sites):
    if(layout_num==6):
        layout_num=0
        plt.figure(figsize=(16, 9))
        plt.suptitle(u'分站点预测/实际值对比（测试数据）')
    subplot = plt.subplot(3, 2, layout_num + 1)
    #site = site_names[i]
    plt.plot(testPredict[:, i])
    plt.plot(test_set_y[:, i])
    plt.xlabel(u'数据编号')
    plt.ylabel(u'数值')
    plt.legend(labels=[u'预测数据',u'实际数据'])
    subplot.set_title(site_cnames[i])
    plt.tight_layout()
    layout_num=layout_num+1

#作图：MSE与MAE
plt.figure(figsize=(16,9))
mseTrain = plt.subplot(221)
maeTrain = plt.subplot(222)
mseTest = plt.subplot(223)
maeTest = plt.subplot(224)

mse=[acc_tr[:,0],acc_t[:,0]]
mse=np.array(mse)
mse=mse.T

mae=[acc_tr[:,1],acc_t[:,1]]
mae = np.array(mae)
mae = mae.T

msemaeList = [mseTrain,maeTrain,mseTest,maeTest]
msemaeData = [mse[:,0],mae[:,0],mse[:,1],mae[:,1]]
msemaeLabels = [u'训练MSE',u'训练MAE',u'测试MSE',u'测试MAE']
def plot4(i):
    plt.plot(msemaeData[i])
    plt.xlabel(u'迭代次数')
    plt.ylabel(u'误差值')
for i in range(0,4):
    plt.sca(msemaeList[i])
    plot4(i)
    plt.legend(labels = [msemaeLabels[i]],loc = 'best')
    msemaeList[i].set_title(msemaeLabels[i])
plt.suptitle(u'训练与测试MSE/MAE对比')

plt.show()

#将预测数据输出为csv文件
months = np.linspace(delay+1,num_Data,num_Data-10)
months = np.array(months)
with open(u"输出文件："+file_name, "w", newline="",encoding="utf-8-sig") as datacsv:
    csvwriter = csv.writer(datacsv, dialect=("excel"))
    first_row = [u'月份/地点']
    for i in range(0,num_sites):
        first_row.append(site_cnames[i]+u'预测')
        first_row.append(site_cnames[i]+u'实际')
    csvwriter.writerow(first_row)
    for i in range(0, trLen):
        train_row = [months[i]]
        for j in range(0, num_sites):
            train_row.append(trainPredict[i, j])
            train_row.append(train_set_y[i, j])
        csvwriter.writerow(train_row)
    for i in range(0, num_Data-trLen-10):
        test_row = [months[i + trLen]]
        for j in range(0,num_sites):
            test_row.append(testPredict[i,j])
            test_row.append(test_set_y[i,j])
        csvwriter.writerow(test_row)
