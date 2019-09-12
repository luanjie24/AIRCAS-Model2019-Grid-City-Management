# -*- coding: utf-8 -*-

#from __future__ import print_function
#from keras.models import Sequential
import keras.backend as K
from keras.callbacks import LearningRateScheduler
import shutil
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
#import random 
import os
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from matplotlib.font_manager import _rebuild
#from sklearn import preprocessing
#如需进行数据归一化则写入下一行代码
from sklearn.preprocessing import MinMaxScaler
#=====================================================================
PARAMS = [[0.001,50,10,1],[0.001,50,10,8] ,[0.001,50,10,32],
          [0.001,10,10,16],[0.001,100,10,16],[0.001,500,10,16]]
for PARAM_NUM in range(7,8):
    #PARAM_NUM = 3   #在这里输入参数号
    NAME = u"无照经营"
    FILE_NAME = NAME+"-所有街道数据.csv"
    F_NAME = u"../图表/LSTM图表/分站点预测/参数"+str(PARAM_NUM)+"/"+NAME+"/"
    #=====================================================================
    #将下面这个参数从1开始跑，
    #比如num_site=1跑一次，num_site=2跑一次，num_site=3跑一次。。。以此类推，直到系统提示“本文件所有站点预测完毕”为止

    #num_site=12


    CURRENT_PARAMS = PARAMS[PARAM_NUM-2]
    #=====================================================================
    #my_lr=0.001      #初始学习率
    my_lr=CURRENT_PARAMS[0]
    #my_node=50       #LSTM层的节点数
    my_node = CURRENT_PARAMS[1]
    #delay=10         #根据前delay个数据预测下一个
    delay=CURRENT_PARAMS[2]
    #my_batch_size=1 #batch_size
    my_batch_size=CURRENT_PARAMS[3]

    #=====================================================================
    #os.makedirs(F_NAME)
    #os.makedirs(u"../LSTM分站点输出预测数据文件/参数" + str(PARAM_NUM) + "/" + NAME + "/")
    #os.makedirs(u'../LSTM分站点结果分析/参数'+str(PARAM_NUM)+u'极端数据图表/')

    df=pd.read_csv(FILE_NAME,encoding='gbk')
    #df=pd.read_csv(FILE_NAME)
    #=================================常量的定义或声明====================================
    NUM_ROW = df.shape[0] #数据行数=表格行数-1（减表头）
    ROW=48 #如果报错尝试这个 将288变为相应的行数
    DATA_SIZE=48 #数据量 每个街道有DATA_SIZE个月的数据
    SITE_NUM=int(NUM_ROW/DATA_SIZE)#站点个数
    SITE_SIZE=1#分站点预测，所以只能等于1，这是最简便的更改方式
    #if(num_site>SITE_NUM):
    #    print("本文件所有站点预测完毕")
    #    exit(0)
    inaccurate_sites = [3,4,7,8,11]
    for num_site in inaccurate_sites:
        for i in range(0,5):
            hourlyData=df.values[DATA_SIZE*(num_site-1):DATA_SIZE*num_site,3]


            hourlyData=hourlyData.astype('float32')#写成科学计数法（float32）
            Mon=df.values[:,2]#获得表格所有月份Mon


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
            scaler = MinMaxScaler(feature_range=(0.01, 1))#这个归一化也有影响 比如要是（0，1）就无法拟合
            #scaler = MinMaxScaler()#这种有的就无法拟合
            hourlyData = scaler.fit_transform(hourlyData)

            #此处应该是将时间序列转换为x,y的监督学习问题
            for d in range(delay,len(hourlyData)-pre+1):
                X_one=hourlyData[d-delay:d,:]#二维
                X_one=X_one.reshape((1,X_one.shape[0],X_one.shape[1]))#转三维 1页1行10列
                y_one=hourlyData[d,:]
                X.append(X_one)
                y.append(y_one)
            X=np.array(X).reshape((len(X),delay,SITE_SIZE)) #reshape页、行、列 三维
            y=np.array(y) #二维
            Mon = np.linspace(delay+1,DATA_SIZE,DATA_SIZE-delay)
            #Mon = np.linspace(1,DATA_SIZE,DATA_SIZE)
            #shuffle data
            index=np.arange(DATA_SIZE-delay)
            np.random.shuffle(index)
            X=X[index,:,:]
            y=y[index]
            Mon=Mon[index]


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
            input_shape = (delay,SITE_SIZE) #每delay个数据预测一个 输入格式为delay行SITE_SIZE列的矩阵
            main_input = Input(shape=input_shape, name='main_input')

            rnn_out = LSTM(my_node, return_sequences=True,consume_less = 'gpu')(main_input)
            x = LSTM(my_node,consume_less = 'gpu')(rnn_out)

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

            solver = Adam(lr=my_lr) #学习率为0.001 一条直线的有可能是学习率过大的缘故
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
            epoches = 80
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
                #reduce_lr = LearningRateScheduler(scheduler)

                if epoch==50:
                    solver = Adam(lr=my_lr/10)#降低学习率
                    model.compile(optimizer=solver,
                              loss={'main_output': 'mape'} )
                hist = model.fit({'main_input': train_set_x},
                              {'main_output': train_set_y},validation_data=(
                                {'main_input': test_set_x, },
                                {'main_output': test_set_y}
                              ),verbose = 1,
                              nb_epoch=10, batch_size=my_batch_size)
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
            #site_cnames.append(df.at[(num_site-1)*DATA_SIZE, u'事发街道'])
            site_cnames.append("站点"+str(num_site))#隐藏站点名称
            site_names.append(hourlyData[:,0])

            mse = [acc_tr[:, 0], acc_t[:, 0]]
            mse = np.array(mse)
            mse = mse.T

            mae = [acc_tr[:, 1], acc_t[:, 1]]
            mae = np.array(mae)
            mae = mae.T

            mape = [acc_tr[:, 2], acc_t[:, 2]]
            mape = np.array(mape)
            mape = mape.T

            if mape[-1, -1] < 0.5:
                if os.path.exists(u'../LSTM分站点结果分析/参数' + str(PARAM_NUM) + u'极端数据图表/' + '【不准】' +
                          NAME + '-站点' + str(num_site) + '-当前打乱的原始数据.png'):
                    os.remove(u'../LSTM分站点结果分析/参数' + str(PARAM_NUM) + u'极端数据图表/' + '【不准】' +
                          NAME + '-站点' + str(num_site) + '-当前打乱的原始数据.png')

                #作图：立案量vs时间
                plt.figure(figsize=(16,9))
                layout_num = 0
                for i in range(0,SITE_SIZE):
                    if(layout_num==1):
                        layout_num=0
                        plt.figure(figsize=(16, 9))
                        plt.suptitle(u'各地点按月立案数量')
                    subplot = plt.subplot(1, 1, layout_num + 1)
                    site = site_names[i]
                    plt.plot(site_names[i])
                    plt.xlabel(u'时间')
                    plt.ylabel(u'立案量')
                    plt.legend(labels=[u'立案量'],loc = 'best')
                    subplot.set_title(str(NAME)+'-'+site_cnames[i]+u"-当前打乱的原始数据")
                    plt.tight_layout()
                    layout_num = layout_num + 1
                plt.savefig(F_NAME+NAME+'-'+str(site_cnames[0])+"-当前打乱的原始数据.png")

                #作图：精度vs迭代次数
                plt.figure(figsize=(16,9))
                #x2 = np.linspace(1,100,len(a[:,0]))
                #x21 = np.linspace(1,100,len(a[:,1]))
                plt.plot(a[:,0])
                plt.plot(a[:,1])
                plt.xlabel(u'迭代次数')
                plt.ylabel(u'精度')
                plt.title(str(NAME)+'-'+str(site_cnames[0])+u'-测试/训练精度与时间对比')
                plt.legend(labels = [u'训练精度',u'测试精度'],loc ='best')
                plt.savefig(F_NAME+NAME+'-'+str(site_cnames[0])+"-精度.png")

                #作图：真实值&预测值vs时间
                #训练数据组

                plt.figure(figsize=(16,9))
                #plt.suptitle(u'分站点预测/实际值对比（训练数据）')
                layout_num = 0
                for i in range(0,SITE_SIZE):
                    if(layout_num==6):
                        layout_num=0
                        plt.figure(figsize=(16, 9))
                        plt.suptitle(u'分站点预测/实际值对比（训练数据）')
                    subplot = plt.subplot(1, 1, layout_num + 1)
                    #site = site_names[i]
                    plt.plot(trainPredict[:, i])
                    plt.plot(train_set_y[:, i])
                    plt.xlabel(u'数据编号')
                    plt.ylabel(u'数值')
                    plt.legend(labels=[u'预测数据',u'实际数据'])
                    subplot.set_title(str(NAME)+'-'+site_cnames[i]+u'-预测/实际值对比（训练数据）')
                    plt.tight_layout()
                    layout_num = layout_num + 1
                plt.savefig(F_NAME+NAME+'-'+str(site_cnames[0])+"-训练预测.png")

                #测试数据组
                plt.figure(figsize=(16,9))
                #plt.suptitle(u'分站点预测/实际值对比（测试数据）')
                layout_num = 0
                for i in range(0,SITE_SIZE):
                    if(layout_num==6):
                        layout_num=0
                        plt.figure(figsize=(16, 9))
                        plt.suptitle(u'分站点预测/实际值对比（测试数据）')
                    subplot = plt.subplot(1, 1, layout_num + 1)
                    #site = site_names[i]
                    plt.plot(testPredict[:, i])
                    plt.plot(test_set_y[:, i])
                    plt.xlabel(u'数据编号')
                    plt.ylabel(u'数值')
                    plt.legend(labels=[u'预测数据',u'实际数据'])
                    subplot.set_title(str(NAME)+'-'+site_cnames[i]+u'-预测/实际值对比（测试数据）')
                    plt.tight_layout()
                    layout_num=layout_num+1
                plt.savefig(F_NAME+NAME+'-'+str(site_cnames[0])+"-测试预测.png")
                #作图：MSE与MAE MAPE
                plt.figure(figsize=(16, 9))
                mseTrain = plt.subplot(321)
                maeTrain = plt.subplot(322)
                mseTest = plt.subplot(323)
                maeTest = plt.subplot(324)

                mapeTrain = plt.subplot(325)
                mapeTest = plt.subplot(326)

                msemaeList = [mseTrain,maeTrain,mseTest,maeTest,mapeTrain,mapeTest]
                msemaeData = [mse[:,0],mae[:,0],mse[:,1],mae[:,1],mape[:,0],mape[:,1]]
                msemaeLabels = [u'训练MSE',u'训练MAE',u'测试MSE',u'测试MAE','训练MAPE','测试MAPE']
                def plot4(i):
                    plt.plot(msemaeData[i])
                    plt.xlabel(u'迭代次数')
                    plt.ylabel(u'误差值')
                for i in range(0,6):
                    plt.sca(msemaeList[i])
                    plot4(i)
                    plt.legend(labels = [msemaeLabels[i]],loc = 'best')
                    msemaeList[i].set_title(str(NAME)+'-'+str(site_cnames[0])+msemaeLabels[i])
                plt.suptitle(str(NAME)+'-'+str(site_cnames[0])+u'-训练与测试MSE/MAE/MAPE对比')
                plt.tight_layout()

                plt.savefig(F_NAME+NAME+'-'+str(site_cnames[0])+"-误差.png")
                #plt.show()
                #将预测数据输出为csv文件
                #months = np.linspace(delay+1,num_Data,num_Data-10)
                months = Mon
                months = np.array(months)

                with open(u"../LSTM分站点输出预测数据文件/参数"+str(PARAM_NUM)+"/"+NAME+"/预测输出："+NAME+"-站点"+str(num_site)+'.csv', "w", newline="",encoding="utf-8-sig") as datacsv:
                    csvwriter = csv.writer(datacsv, dialect=("excel"))
                    first_row = [u'月份/地点']
                    for i in range(0,SITE_SIZE):
                        first_row.append(site_cnames[i]+u'预测')
                        first_row.append(site_cnames[i]+u'实际')
                    csvwriter.writerow(first_row)
                    for i in range(0, trLen):
                        train_row = [months[i]]
                        for j in range(0, SITE_SIZE):
                            train_row.append(trainPredict[i, j])
                            train_row.append(train_set_y[i, j])
                        csvwriter.writerow(train_row)
                    for i in range(0, DATA_SIZE-trLen-delay):
                        test_row = [months[i + trLen]]
                        for j in range(0,SITE_SIZE):
                            test_row.append(testPredict[i,j])
                            test_row.append(test_set_y[i,j])
                        csvwriter.writerow(test_row)
                break




