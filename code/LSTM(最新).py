# -*- coding: utf-8 -*-

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


FILE_NAME = u"非法小广告-所有街道数据.csv"
df=pd.read_csv(FILE_NAME,encoding='gbk')
#df=pd.read_csv(FILE_NAME)
#=================================常量的定义或声明====================================
#ROW = df.shape[0] #数据行数=表格行数-1（减表头）
ROW=48 #如果报错尝试这个 将288变为相应的行数
DATA_SIZE=48 #数据量 每个街道有DATA_SIZE个月的数据
SITE_SIZE=int(ROW/DATA_SIZE)#站点个数
SITE_SIZE=1

hourlyData=df.values[:ROW,3]
hourlyData=hourlyData.astype('float32')#写成科学计数法（float32）
Mon=df.values[:,2]#获得表格所有月份Mon

delay=10 #根据前delay个数据预测下一个
X=[] #输入 根据delay个数据成一组作为X，得到输出y （一个X有delay个数据，有多组X，所以是二维的,再加上多个街道，变成三维的了）
y=[] #输出
pre=1 #不知道这是干啥的

#设置绘图时的中文显示（需安装黑体字体）
_rebuild()
mpl.rcParams['font.sans-serif']=[u'SimHei']
mpl.rcParams['axes.unicode_minus']=False
#=================================================================================

#输出一下得到的案件量个数
print('aqi data length:', len(hourlyData))

#立案量，转换成SITE_SIZE行DATA_SIZE列，每行代表不同的街道，每个街道有DATA_SIZE个月的数据
hourlyData = hourlyData.reshape(SITE_SIZE,DATA_SIZE)

#转置 现在每一列代表不同的街道了
hourlyData = hourlyData.T

#====================================以下为对训练样本的预处理 归一化、打乱、分训练测试组等================================
#在进行运算之前可以对数据进行归一化，进而降低loss
scaler = MinMaxScaler(feature_range=(0.001, 1))
hourlyData = scaler.fit_transform(hourlyData)

#此处应该是将时间序列转换为x,y的监督学习问题
for d in range(delay,len(hourlyData)-pre+1):
    X_one=hourlyData[d-delay:d,:]
    X_one=X_one.reshape((1,X_one.shape[0],X_one.shape[1]))
    y_one=hourlyData[d,:]
    X.append(X_one)
    y.append(y_one)
X=np.array(X).reshape((len(X),delay,SITE_SIZE)) #reshape页、行、列 三维
y=np.array(y) #二维

'''
print("X")
print(X)
print("y")
print(y)
'''

#shuffle data
#随机排列x,y,Mon，但一一对应
random.seed(10)
random.shuffle(X)
random.seed(10)
random.shuffle(y)
random.seed(10)
random.shuffle(Mon)

#split dataset
#将数据分成训练组和测试组 前80%的数据作为训练，后20%的数据作为测试
trLen=int(0.8*X.shape[0])
train_set_x=X[:trLen,:]
train_set_y=y[:trLen]
test_set_x = X[trLen:,:]
test_set_y=y[trLen:]

'''
print("train_set_x")
print(train_set_x)
print("train_set_y")
print(train_set_y)
print("test_set_x")
print(test_set_x)
print("test_set_y")
print(test_set_y)
'''
#====================================================================

#==========================================本模块采用LSTM建模================================================
# build the model: 2 stacked LSTM
print('Build model...')
input_shape = (delay,SITE_SIZE) #每delay个数据预测一个 输入格式为delay行SITE_SIZE列的矩阵
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
loss = Dense(SITE_SIZE, activation='relu', name='main_output')(x)

#使用刚才创建的图生成模型
model = Model(input=[main_input], output=[loss])

solver = Adam(lr=0.001) #学习率为0.001
model.compile(optimizer=solver,
                  loss={'main_output': 'mape'} ) #optimizer优化器选择Adam 回头可以再尝试一下RMSprop
                                                #损失函数loss用的mape?

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

#把模型写入jason文件中，权重记录在.hdf5中？因为每次的权中事随机的 
model_json = model.to_json()
model_path = '$8.json'
model_weight_path = '$8_weights.hdf5'
with open(model_path, "w") as json_file:
    json_file.write(model_json)

#迭代次数为100次
epoches = 60
#生成epoches行4列的零矩阵
acc_tr=np.zeros((epoches,4))
acc_t=np.zeros((epoches,4))
history = []

#生成epoches行2列的零矩阵
#msemae_tr = np.zeros((epoches,2))
#msemae_t = np.zeros((epoches,2))

#================================================================训练LSTM=====================================
#开始迭代
for epoch in range(epoches):
    print()
    print('-' * 50)
    print('epoch', epoch)
    if epoch==50:
        solver = Adam(lr=0.0001)#降低学习率
        model.compile(optimizer=solver,
                  loss={'main_output': 'mape'} ) 
    hist = model.fit({'main_input': train_set_x},
                  {'main_output': train_set_y},validation_data=(
                    {'main_input': test_set_x, },
                    {'main_output': test_set_y}
                  ),verbose = 1,
                  nb_epoch=10, batch_size=int(DATA_SIZE/2)) #batch_size待确定 nb_epoch是训练数据遍历的次数
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
for num in range(0,SITE_SIZE):
    site_cnames.append(df.at[num*DATA_SIZE, u'事发街道'])
    site_names.append(hourlyData[:,num])



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
for i in range(0,SITE_SIZE):
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
for i in range(0,SITE_SIZE):
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
plt.figure(5)
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
months = np.linspace(delay+1,DATA_SIZE,DATA_SIZE-10)
months = np.array(months)
with open(u"输出文件："+FILE_NAME, "w", newline="",encoding="utf-8-sig") as datacsv:
    csvwriter = csv.writer(datacsv, dialect=("excel"))
    first_row = [u'月份/地点']
    for i in range(0,SITE_SIZE):
        first_row.append(site_cnames[i]+u'预测')
        first_row.append(site_cnames[i]+u'实际')
    csvwriter.writerow(first_row)
    #csvwriter.writerow([u'月份/地点', legend_labels[0],legend_labels[6],legend_labels[1],legend_labels[7],
    #                   legend_labels[2],legend_labels[8],legend_labels[3],legend_labels[9],legend_labels[4],
    #                   legend_labels[10],legend_labels[5],legend_labels[11]])
    for i in range(0, trLen):
        train_row = [months[i]]
        for j in range(0, SITE_SIZE - 1):
            train_row.append(trainPredict[i, j])
            train_row.append(trainPredict[i, j])
        csvwriter.writerow(train_row)
        #csvwriter.writerow([months[i], trainPredict[i, 0],train_set_y[i,0], trainPredict[i, 1], train_set_y[i,1],
        #                    trainPredict[i, 2], train_set_y[i,2], trainPredict[i, 3], train_set_y[i,3],
        #                    trainPredict[i, 4],train_set_y[i,4], trainPredict[i, 5],train_set_y[i,5]])
    for i in range(0, DATA_SIZE-trLen-10):
        test_row = [months[i + trLen]]
        for j in range(0,SITE_SIZE-1):
            test_row.append(testPredict[i,j])
            test_row.append(testPredict[i,j])
        csvwriter.writerow(test_row)
        #csvwriter.writerow([months[i + trLen], testPredict[i, 0],test_set_y[i,0], testPredict[i, 1],test_set_y[i,1],
        #                   testPredict[i, 2],test_set_y[i,2], testPredict[i, 3],test_set_y[i,3],
        #                  testPredict[i, 4],test_set_y[i,4],testPredict[i, 5],test_set_y[i,5]])

