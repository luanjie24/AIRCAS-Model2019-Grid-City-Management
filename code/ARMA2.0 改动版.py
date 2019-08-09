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
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from matplotlib.font_manager import _rebuild
from sklearn.preprocessing import MinMaxScaler

#设置绘图时的中文显示（需安装黑体字体）
_rebuild()
mpl.rcParams['font.sans-serif']=[u'SimHei']
mpl.rcParams['axes.unicode_minus']=False

#读取数据
FILE_NAME=u"暴露垃圾-所有街道数据.csv"
df=pd.read_csv(FILE_NAME,encoding="gbk")

#=================================需更新的常量====================================
ROW = df.shape[0] #数据行数=表格行数-1（减表头）
COLUMN = 9 #数据列数=表格列数
DATA_SIZE= 48 #数据量 每个街道有DATA_SIZE个月的数据
inData=df.values[:ROW,3]
inData=inData.astype('float32')
SITE_SIZE=int(ROW/DATA_SIZE)
delay = 10
#=================================================================================

#在进行运算之前可以对数据进行归一化，进而降低loss
scaler = MinMaxScaler()
inData = scaler.fit_transform(inData.reshape(-1,1))
#=================================================================================
#数据平稳化，使Adfuller指数小于-3.435298，并且p-value小于0.05
ADF = adfuller(inData.ravel())
print("ADF:")
print(ADF)

stattools.q_stat(stattools.acf(inData)[1:13],len(inData))[1][-1]
plot_acf(inData, lags= 30)
plot_pacf(inData, lags= 30)

#定阶ARMA(p,q)
order = stattools.arma_order_select_ic(inData,max_ar=3,max_ma=3,ic=['aic','bic','hqic'])
print("(p,q):")
print(order)
pq = order.bic_min_order
print(order.bic_min_order)#(p,q)

#将数据按站点分为6组
#site_names=[]  #站点数据列表
site_cnames=[] #站点名字列表
for num in range(0,SITE_SIZE):
    site_cnames.append(df.at[num*DATA_SIZE, u'事发街道'])
    '''if num==0:
        site_names.append(inData[1:DATA_SIZE-1])  
    else:
        site_names.append(inData[num*DATA_SIZE:(num+1)*DATA_SIZE-1])
site_names = np.array(site_names)'''
site_names = inData.reshape(SITE_SIZE,DATA_SIZE)
site_names = site_names.T
#拟合（生成训练模型），开始预测
plt.figure(3)
plt.suptitle(u'分站点预测/实际值对比')
MSE = []
MAE = []
MAPE = []
#model = ARMA(inData, order.bic_min_order).fit()
for i in range(0,SITE_SIZE):
    
    subplot = plt.subplot(6,4,i+1)
    site = site_names[:,i]
    model = ARMA(site, (3,0)).fit()
    #model = ARMA(site,(3,0)).fit()
    predict_data = model.predict(start=0,end=DATA_SIZE-1)

    #在这里进行反归一化#===============
    predict_data = scaler.inverse_transform(predict_data.reshape(-1,1))#====================
    site = scaler.inverse_transform(site.reshape(-1,1))#==============

    plt.plot(predict_data)
    plt.plot(site)
    plt.xlabel(u'时间')
    plt.ylabel(u'数值')
    plt.legend(labels = [u'预测数据',u'实际数据'])
    subplot.set_title(site_cnames[i])
    #MSE与MAE
    error = []
    pctError = []
    
    #print(predict_data)
    print("123done")
    #predict_data = scaler.inverse_transform(predict_data.reshape(-1,1))
    #print(predict_data)
    
    for j in range(len(site)):
        #site[j] = scaler.inverse_transform(site[j].reshape(-1,1))
        #上面已经进行了反归一化。重点在于site 和 predict data需要在作图之前反归一化
        error.append(predict_data[j]-site[j])
        pctError.append(abs((predict_data[j]-site[j])/site[j]))
    sqError = []
    absError = []
    for val in error:
        sqError.append(val*val)
        absError.append(abs(val))
    mse = sum(sqError)/len(sqError)
    mae = sum(absError)/len(absError)
    mape = sum(pctError)/len(pctError)
    MAPE.append(mape)
    MSE.append(mse)
    MAE.append(mae)
    print('mse =',mse,'mae=',mae)
#plot MSE,MAE
plt.figure(4)
msep = plt.subplot(211)
maep = plt.subplot(212)
plt.sca(msep)
plt.plot(site_cnames,MSE)
plt.xlabel(u'地点名称')
plt.ylabel(u'误差')
plt.legend(labels = [u'MSE'])
msep.set_title(u'MSE')
plt.sca(maep)
plt.plot(site_cnames,MAE)
plt.xlabel(u'地点名称')
plt.ylabel(u'误差')
plt.legend(labels = [u'MAE'])
maep.set_title(u'MAE')
plt.suptitle(u'分站点MSE与MAE对比')
#plot MAPE
plt.figure(5)
plt.plot(site_cnames, MAPE)
plt.xlabel(u'地点名称')
plt.ylabel(u'误差比')
plt.legend(labels = ['MAPE'])
plt.title(u'分站点MAPE对比')

plt.show()
