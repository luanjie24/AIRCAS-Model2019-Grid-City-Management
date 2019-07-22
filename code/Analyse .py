#本程序用于分析原有数据，或者经过处理的原有数据
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
K=2 #K为差分阶数
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

diff_df = np.log(df["立案量"])#差分前对数平滑处理
diff_df=diff_df.diff(K)#差分用的
diff_df=diff_df.tolist()

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
diff_data=[]  #差分数据列表
for num in range(0,SITE_SIZE):
    site_names.append(df.at[num*DATA_SIZE, u'事发街道'])
    if num==0:
        site_data.append(inData[1:DATA_SIZE-1])
        diff_data.append(diff_df[1:DATA_SIZE-1])
    else:
        site_data.append(inData[num*DATA_SIZE:(num+1)*DATA_SIZE-1])
        diff_data.append(diff_df[num*DATA_SIZE:(num+1)*DATA_SIZE-1])



#分析样本ADF：
def show_ADF():
    for i in range(0,SITE_SIZE):
        site = site_data[i]
        name=site_names[i]
        ADF = adfuller(site.ravel(),1)
        print(name+"ADF:")
        print(ADF)

def show_diff_ADF(diff_data):
    for i in range(0,SITE_SIZE):
        site = diff_data[i]
        name=site_names[i]
        ADF = adfuller(site.ravel(),1)
        print(name+"ADF:")
        print(ADF)




#对样本求移动平均值并分析趋势
def mean_it():
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


#对样本做差分操作并分析趋势
def diff_it():
    layout_num3=0#画图排版用的

    #将数据按站点分为SITE_SIZE组
    diff_data=[]  #站点数据列表
    diff_names=[] #站点名字列表
    for num in range(0,SITE_SIZE):
        diff_names.append(df.at[num*DATA_SIZE, u'事发街道'])
        #选取当前街道的所有立案量
        temp=df.loc[(df[u'事发街道']==diff_names[num]) ,[u'立案量']] 
        #K阶差分
        temp=temp.diff(K)
        #去掉缺失值
        #temp=temp.where(temp.notnull(), 0.00001)
        temp=temp.dropna()
        diff_data.append(temp.values)

    
    for i in range(0,SITE_SIZE):
        diff_site=diff_data[i]
        if(layout_num3==6):
            layout_num3=0
            plt.figure(figsize=(16, 9))
        
        subplot = plt.subplot(3,2,layout_num3+1)
        #画数据
        #plt.plot(site,color='yellow',label='原始数据') #缺省x为[0,1,2,3,4,...]
        #画差分后的数据
        plt.plot(diff_site,color='blue',label='差分数据') #缺省x为[0,1,2,3,4,...]

        plt.xlabel(u'时间')
        plt.ylabel(u'立案量')
        plt.legend(loc='best')
        subplot.set_title(diff_names[i])
        plt.tight_layout()

        layout_num3=layout_num3+1

        
    show_diff_ADF(diff_data)
    plt.show()



#对样本做差分操作并分析趋势
def diff_log_it():
    layout_num3=0#画图排版用的

    #将数据按站点分为SITE_SIZE组
    diff_data=[]  #站点数据列表
    diff_names=[] #站点名字列表
    for num in range(0,SITE_SIZE):
        
        diff_names.append(df.at[num*DATA_SIZE, u'事发街道'])
        #选取当前街道的所有立案量
        temp=df.loc[(df[u'事发街道']==diff_names[num]) ,[u'立案量']] 
        #对数化
        temp = np.log(temp)
        #K阶差分
        temp=temp.diff(K)
        #去掉缺失值
        #temp=temp.where(temp.notnull(), 0.00001)
        temp=temp.dropna()
        diff_data.append(temp.values)

    
    for i in range(0,SITE_SIZE):
        site=diff_data[i]
        diff_site=diff_data[i]
        if(layout_num3==6):
            layout_num3=0
            plt.figure(figsize=(16, 9))
        
        subplot = plt.subplot(3,2,layout_num3+1)
        #画数据
        #plt.plot(site,color='yellow',label='原始数据') #缺省x为[0,1,2,3,4,...]
        #画差分后的数据
        plt.plot(diff_site,color='blue',label='对数化并差分数据') #缺省x为[0,1,2,3,4,...]

        plt.xlabel(u'时间')
        plt.ylabel(u'立案量')
        plt.legend(loc='best')
        subplot.set_title(diff_names[i])
        plt.tight_layout()

        layout_num3=layout_num3+1
    show_diff_ADF(diff_data)
    plt.show()



#对样本序列进行平稳化操作并分析数据趋势
def log_it():
    layout_num=0#画图排版用的
    #定义移动窗口和权重
    N=6
    n=np.ones(N)
    weights=n/N
    plt.figure(figsize=(12, 8))
    #plt.suptitle(u'处理后数据趋势分析图')
    for i in range(0,SITE_SIZE):
        site = site_data[i]
        #对数化，让序列更光滑
        log_site = np.log(site)
        #调用convolve函数，并从获得的数组中取出长度为N的部分
        sma=np.convolve(weights,log_site)[N-1:-N+1]
        t=np.arange(N-1,len(log_site))
        #做差
        site_temp=log_site[N-1:]
        sma_temp=sma
        diff = site_temp-sma_temp
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
    mean_it()
    #show_ADF()
    #log_it()
    #diff_it()
    #diff_log_it()

    
