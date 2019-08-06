#本程序用于分析原有数据
import pandas as pd
import numpy as np
from scipy import stats
import tushare as ts
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
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
FILE_NAME=u"无照经营-所有街道数据.csv"
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


#decomposition = seasonal_decompose(df["立案量"], model="additive")



#案件数据
inData=df.values[:ROW,3]

#inData = np.log(inData)


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

'''
ts_log = np.log(inData)
f = plt.figure(facecolor='white')
inData.plot(color='blue')
plt.show()
'''


#分析原始样本趋势
def show_means():
    layout_num2=0#画图排版用的
    #定义移动窗口和权重
    N=6
    n=np.ones(N)
    weights=n/N
    plt.figure(figsize=(12, 8))
    plt.suptitle(u'原始样本数据趋势分析图')
    for i in range(0,SITE_SIZE):
        if(layout_num2==6):
            layout_num2=0
            plt.figure(figsize=(16, 9))
        subplot = plt.subplot(3,2,layout_num2+1)
        site = site_data[i]
        #画数据
        plt.plot(site,color='blue',label='原始数据') #缺省x为[0,1,2,3,4,...]

        #画移动平均值
        #调用convolve函数，并从获得的数组中取出长度为N的部分
        sma=np.convolve(weights,site)[N-1:-N+1]
        t=np.arange(N-1,len(site))
        plt.plot(t,sma,lw=2,color='red', label='移动平均值')
        
        plt.xlabel(u'时间')
        plt.ylabel(u'数值')
        plt.legend(loc='best')
        subplot.set_title(site_names[i])
        plt.tight_layout()

        layout_num2=layout_num2+1
    plt.show()  




#分析样本ADF：
def show_ADF():
    for i in range(0,SITE_SIZE):
        site = site_data[i]
        name=site_names[i]
        ADF = adfuller(site.ravel(),1)
        print(name+"ADF:")
        print(ADF)




#对样本序列进行平稳化操作并分析数据趋势
def analyse_data():
    layout_num=0#画图排版用的
    #定义移动窗口和权重
    N=6
    n=np.ones(N)
    weights=n/N
    plt.figure(figsize=(12, 8))
    '''
    #移动平均值的开始横坐标
    x=[]
    for j in range(1,DATA_SIZE-1):
        x.append(j)
    print(x)
    '''
    #plt.suptitle(u'处理后数据趋势分析图')
    for i in range(0,SITE_SIZE):
        
        
        site = site_data[i]

        #对数化，让序列更光滑
        log_site = np.log(site)

        #print(log_site)

        #调用convolve函数，并从获得的数组中取出长度为N的部分
        sma=np.convolve(weights,log_site)[N-1:-N+1]
        t=np.arange(N-1,len(log_site))
        #做差
        site_temp=log_site[N-1:]
        sma_temp=sma
        diff = site_temp-sma_temp
        #print(site)
        
        if(layout_num==6):
            layout_num=0
            plt.figure(figsize=(16, 9))
        
        subplot = plt.subplot(3,2,layout_num+1)
        #画对数化以后的数据
        plt.plot(log_site,color='yellow',label='对数化数据') #缺省x为[0,1,2,3,4,...]
        #画移动平均值
        plt.plot(t,sma,lw=2,color='red', label='对数化后的移动平均值')
        #画差
        plt.plot(t,diff,lw=2,color='green', label='对数化后作差')




        plt.xlabel(u'时间')
        plt.ylabel(u'立案量')
        plt.legend(loc='best')
        subplot.set_title(site_names[i])
        plt.tight_layout()

        layout_num=layout_num+1

    plt.show()  


if __name__ == '__main__':
    #show_means()
    #show_ADF()
    analyse_data()

    
