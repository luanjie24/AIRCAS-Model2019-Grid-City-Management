import pandas as pd
import numpy as np
from scipy import stats
import tushare as ts
import datetime
#from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.api import qqplot
from statsmodels.tsa import stattools
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA

import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from matplotlib.font_manager import _rebuild
from sklearn.preprocessing import MinMaxScaler

#=================================需更新的常量====================================
ROW = 288 #数据行数=表格行数-1（减表头） 这个现在在后面自动读取，不用改了 但有时候会出错
COLUMN = 9 #数据列数=表格列数 这个通常不用改
DATA_SIZE= 48 #数据量 每个街道有DATA_SIZE个月的数据 这个通常不同改
FILE_NAME=u"积存渣土-所有街道数据.csv"
#=================================================================================

#设置绘图时的中文显示（需安装黑体字体）
_rebuild()
mpl.rcParams['font.sans-serif']=[u'SimHei']
mpl.rcParams['axes.unicode_minus']=False

#读取数据
df=pd.read_csv(FILE_NAME,encoding="gbk")
#df=pd.read_csv(FILE_NAME)
print("row:")
print(len(df))
ROW=len(df)
SITE_SIZE=int(ROW/DATA_SIZE)
#案件数据
inData=df.values[:ROW,3]

month=[1,2,3,4,5,6,7,8,9,10,11,12]
try:
    inData=inData.astype('float32')
except:
    print("+++++++++++++++++++++++++++++++++++++++++++++++")
    print("行数row不准确，把ROW=len(df)注释掉改为手动输入行数")
    print("+++++++++++++++++++++++++++++++++++++++++++++++")
#将数据按站点分组
site_data=[]  #站点数据列表
site_names=[] #站点名字列表
for num in range(0,SITE_SIZE):
    site_names.append(df.at[num*DATA_SIZE, u'事发街道'])
    if num==0:
        site_data.append(inData[1:DATA_SIZE-1])  
    else:
        site_data.append(inData[num*DATA_SIZE:(num+1)*DATA_SIZE-1])


#分析原始样本趋势
def show_data():
    plt.figure(figsize=(12, 8))
    plt.suptitle(u'原始样本数据趋势分析图')
    for i in range(0,SITE_SIZE):
        subplot = plt.subplot(4,4,i+1)
        site = site_data[i]
        plt.plot(site) #缺省x为[0,1,2,3,4,...]
        plt.xlabel(u'时间')
        plt.ylabel(u'数值')
        subplot.set_title(site_names[i])
        plt.tight_layout()
    plt.show()  


def test_stationarity(timeseries):

    #Determing rolling statistics
    site = site_data[1]
    rolmean=df.rolling(12,inData).mean()
    #rolmean = site.rolling(1).mean()
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    '''
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput 
    '''

#移动平均值 待修改
def show_means():
    plt.figure(1)
    site = site_data[0]
    N=2
    n=np.ones(N)
    weights=n/N
    sma=np.convolve(weights,site)[N-1:-N+1]
    t=np.arange(N-1,len(site))
    plt.plot(t,site[N-1:],lw=1)
    plt.plot(t,sma,lw=2)
    plt.show()


if __name__ == '__main__':
    #test_stationarity(inData)
    show_data()
    show_means()
    
