import pandas as pd
import numpy as np
from scipy import stats
import tushare as ts
import datetime
from statsmodels.graphics.api import qqplot
from statsmodels.tsa import stattools
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from matplotlib.font_manager import _rebuild
from sklearn.preprocessing import MinMaxScaler

#读取数据
FILE_NAME=u"无照经营-所有街道数据.csv"
df=pd.read_csv(FILE_NAME,encoding="gbk")


#=================================需更新的常量====================================
ROW = df.shape[0] #数据行数=表格行数-1（减表头）
COLUMN = 9 #数据列数=表格列数
DATA_SIZE= 48 #数据量 每个街道有DATA_SIZE个月的数据
inData=df.values[:ROW,3]
inData=inData.astype('float32')
#=================================================================================

SITE_SIZE=int(ROW/DATA_SIZE)

#设置绘图时的中文显示（需安装黑体字体）
_rebuild()
mpl.rcParams['font.sans-serif']=[u'SimHei']
mpl.rcParams['axes.unicode_minus']=False

#取对数以增加平稳性
inData = np.log(inData)#––––––––––––––––––

#在进行运算之前可以对数据进行归一化，进而降低loss
scaler = MinMaxScaler()
#scaler = MinMaxScaler(feature_range=(0,1))
inData = scaler.fit_transform(inData.reshape(-1,1))

'''
#数据平稳化，使Adfuller指数小于-3.435298，并且p-value小于0.05
ADF = adfuller(inData.ravel(),1)
print("ADF:")
print(ADF)
'''

stattools.q_stat(stattools.acf(inData)[1:13],len(inData))[1][-1]
plot_acf(inData, lags= 30)
plot_pacf(inData, lags= 30)

#定阶ARMA(p,q)
'''order = stattools.arma_order_select_ic(inData,max_ar=3,max_ma=3,ic=['aic','bic','hqic'])
print("(p,q):")
pq = order.bic_min_order
print(order.bic_min_order)#(p,q)'''

#将数据按站点分为SITE_SIZE组
site_names=[]  #站点数据列表
site_cnames=[] #站点名字列表
for num in range(0,SITE_SIZE):
    #site_cnames.append(df.at[num*DATA_SIZE, u'事发街道'])
    site_cnames.append("站点"+str(num+1))#隐藏站点名称
    if num==0:
        site_names.append(inData[0:DATA_SIZE])
    else:
        site_names.append(inData[num*DATA_SIZE:(num+1)*DATA_SIZE])

#拟合（生成训练模型），开始预测
plt.figure()
plt.suptitle(u'分站点预测/实际值对比')
MSE = []
MAE = []
MAPE = []
MAPE2 = []
layout_num=0#画图排版用的
for i in range(0,SITE_SIZE):
    if(layout_num==6):
        layout_num=0
        plt.figure(figsize=(16, 9))
        plt.suptitle(u'分站点预测/实际值对比')

    subplot = plt.subplot(3,2,layout_num+1)

    site = site_names[i]
    order = stattools.arma_order_select_ic(site, max_ar=3, max_ma=3, ic=['aic', 'bic', 'hqic'])
    print("(p,q):")
    pq = order.bic_min_order
    print(order.bic_min_order)  # (p,q)

    model = ARMA(site,pq).fit()
    predict_data = model.predict(start=1,end=DATA_SIZE) 
    predict_data2 = model.predict() #0到48
    #model = ARIMA(site, order=(2, 1, 2)).fit()
    #predict_data = model.predict(start=0,end=DATA_SIZE) #0到49

    #在这里进行反归一化#
    predict_data = scaler.inverse_transform(predict_data.reshape(-1,1))
    predict_data2= scaler.inverse_transform(predict_data2.reshape(-1,1))
    site = scaler.inverse_transform(site.reshape(-1,1))
    site = np.exp(site)#–––––––––––––
    predict_data = np.exp(predict_data)#–––––––––––––––
    predict_data2 = np.exp(predict_data2)
    plt.plot(predict_data,color='orange')
    plt.plot(site,color='black')
    plt.xlabel(u'时间')
    plt.ylabel(u'立案量')
    plt.legend(labels = [u'预测数据',u'实际数据'],loc=1)
    subplot.set_title(site_cnames[i])
    plt.tight_layout()
    #MSE与MAE
    error = []
    pctError = []
    error2 = []
    pctError2 = []
    
    #print(predict_data)
    #predict_data = scaler.inverse_transform(predict_data.reshape(-1,1))
    #print(predict_data)
    
    layout_num=layout_num+1
    for j in range(len(site)):
        #site[j] = scaler.inverse_transform(site[j].reshape(-1,1))
        #上面已经进行了反归一化。重点在于site 和 predict data需要在作图之前反归一化
        error.append(predict_data[j]-site[j])
        error2.append(predict_data2[j]-site[j])
        pctError.append(abs((predict_data[j]-site[j])/site[j]))
        pctError2.append(abs((predict_data2[j]-site[j])/site[j]))
    sqError = []
    absError = []
    for val in error:
        sqError.append(val*val)
        absError.append(abs(val))
    mse = sum(sqError)/len(sqError)
    mae = sum(absError)/len(absError)
    mape = sum(pctError)/len(pctError)
    mape2 = sum(pctError2)/len(pctError2)
    MAPE.append(mape)
    MAPE2.append(mape2)
    MSE.append(mse)
    MAE.append(mae)
    print('mse =',mse,'mae=',mae)
#plot MSE,MAE
plt.figure()
msep = plt.subplot(211)
maep = plt.subplot(212)
plt.sca(msep)
plt.plot(site_cnames,MSE)
plt.xlabel(u'站点编号')
plt.ylabel(u'误差')
plt.legend(labels = [u'MSE'],loc=1)
msep.set_title(u'MSE')
plt.sca(maep)
plt.plot(site_cnames,MAE)
plt.xlabel(u'站点编号')
plt.ylabel(u'误差')
plt.legend(labels = [u'MAE'],loc=1)
maep.set_title(u'MAE')
plt.suptitle(u'分站点MSE与MAE对比')
#plot MAPE
plt.figure()
plt.plot(site_cnames, MAPE,color='black')
plt.plot(site_cnames, MAPE2,color='orange')
plt.xlabel(u'站点编号')
plt.ylabel(u'误差比')
plt.legend(labels = ['解决滞后性后的MAPE','解决滞后性前的MAPE'],loc=1)
plt.title(u'分站点MAPE对比')

plt.show()
print("=================MAPE==========")
print(MAPE)
