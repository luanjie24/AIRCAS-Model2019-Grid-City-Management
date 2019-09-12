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
K=2 #K为差分阶数
#=================================================================================

SITE_SIZE=int(ROW/DATA_SIZE)

#设置绘图时的中文显示（需安装黑体字体）
_rebuild()
mpl.rcParams['font.sans-serif']=[u'SimHei']
mpl.rcParams['axes.unicode_minus']=False

#diff_df = np.log(df["立案量"])#差分前对数平滑处理

#取对数以增加平稳性
df[u'立案量'] = np.log(df[u'立案量'])

df[u'立案量']=df[u'立案量'].astype('float32')
#在进行运算之前可以对数据进行归一化，进而降低loss
scaler = MinMaxScaler()
#scaler = MinMaxScaler(feature_range=(0,1))
df[u'立案量'] = scaler.fit_transform(df[u'立案量'].values.reshape(-1,1))

'''
stattools.q_stat(stattools.acf(df[u'立案量'])[1:13],len(df[u'立案量']))[1][-1]
plot_acf(df[u'立案量'], lags= 30)
plot_pacf(df[u'立案量'], lags= 30)
'''

#将数据按站点分为SITE_SIZE组
site_names=[]  #站点数据列表
site_cnames=[] #站点名字列表
for num in range(0,SITE_SIZE):
    site_cnames.append(df.at[num*DATA_SIZE, u'事发街道'])
    #选取当前街道的所有立案量
    temp=df.loc[(df[u'事发街道']==site_cnames[num]) ,[u'立案量']] 
    #K阶差分
    temp=temp.diff(K)
    #去掉缺失值
    #temp=temp.where(temp.notnull(), 0.00001)
    temp=temp.dropna()
    site_names.append(temp.values)

#拟合（生成训练模型），开始预测
plt.figure()
plt.suptitle(u'分站点预测/实际值对比')
MSE = []
MAE = []
MAPE = []
layout_num=0#画图排版用的
for i in range(0,SITE_SIZE):

    if(layout_num==6):
        layout_num=0
        plt.figure(figsize=(16, 9))
        plt.suptitle(u'分站点预测/实际值对比')

    subplot = plt.subplot(3,2,layout_num+1)

    site = site_names[i]
    order = stattools.arma_order_select_ic(site, max_ar=20, max_ma=20, ic=['aic', 'bic', 'hqic'])
    print("(p,q):")
    pq = order.bic_min_order
    print(order.bic_min_order)  # (p,q)



    '''
    AIC = sm.tsa.arma_order_select_ic(timeseries,\
        max_ar=6,max_ma=4,ic='aic')['aic_min_order']
    #BIC
    BIC = sm.tsa.arma_order_select_ic(timeseries,max_ar=6,\
           max_ma=4,ic='bic')['bic_min_order']
    #HQIC
    HQIC = sm.tsa.arma_order_select_ic(timeseries,max_ar=6,\
                 max_ma=4,ic='hqic')['hqic_min_order']
    print('the AIC is{},\nthe BIC is{}\n the HQIC is{}'.format(AIC,BIC,HQIC))

    '''

    model = ARIMA(site, order=(pq[0], pq[1], K)).fit()
    predict_data = model.predict()

    #在这里进行反归一化#
    predict_data = scaler.inverse_transform(predict_data.reshape(-1,1))
    site = scaler.inverse_transform(site.reshape(-1,1))
    site = np.exp(site)#–––––––––––––
    predict_data = np.exp(predict_data)#–––––––––––––––

    plt.plot(predict_data)
    plt.plot(site)
    plt.xlabel(u'时间')
    plt.ylabel(u'立案量')
    plt.legend(labels = [u'预测数据',u'实际数据'])
    subplot.set_title(site_cnames[i])
    plt.tight_layout()
    #MSE与MAE
    error = []
    pctError = []
    
    #print(predict_data)
    #predict_data = scaler.inverse_transform(predict_data.reshape(-1,1))
    #print(predict_data)
    
    layout_num=layout_num+1
    for j in range(len(site)-1):
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
plt.figure()
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
plt.figure()
plt.plot(site_cnames, MAPE)
plt.xlabel(u'地点名称')
plt.ylabel(u'误差比')
plt.legend(labels = ['MAPE'])
plt.title(u'分站点MAPE对比')

plt.show()
