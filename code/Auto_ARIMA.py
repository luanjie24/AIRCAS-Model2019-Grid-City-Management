import pandas as pd
import numpy as np
from scipy import stats
import tushare as ts
import datetime
from statsmodels.graphics.api import qqplot
from statsmodels.tsa import stattools
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from matplotlib.font_manager import _rebuild
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error
#from pyramid.arima import auto_arima
from pmdarima.arima import auto_arima


#load the data

FILE_NAME=u"无照经营-所有街道数据.csv"
df=pd.read_csv(FILE_NAME,encoding="gbk")

ROW = df.shape[0] #数据行数=表格行数-1（减表头）
COLUMN = 9 #数据列数=表格列数
DATA_SIZE= 48 #数据量 每个街道有DATA_SIZE个月的数据
SITE_SIZE=int(ROW/DATA_SIZE)

#设置绘图时的中文显示（需安装黑体字体）
_rebuild()
mpl.rcParams['font.sans-serif']=[u'SimHei']
mpl.rcParams['axes.unicode_minus']=False

#将数据按站点分为SITE_SIZE组
site_data=[]  #站点数据列表
site_names=[] #站点名字列表
temp_name=[]

for num in range(0,SITE_SIZE):
        
    temp_name.append(df.at[num*DATA_SIZE, u'事发街道'])
    site_names.append("站点"+str(num+1))#隐藏站点名称
    #选取当前街道的所有立案量
    temp=df.loc[(df[u'事发街道']==temp_name[num]) ,[u'立案量']] 
    site_data.append(temp)
    
    



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
        plt.figure()
        plt.suptitle(u'分站点预测/实际值对比')

    subplot = plt.subplot(3,2,layout_num+1)
    site = site_data[i]

    #building the model
    model=auto_arima(site)
    model.fit(site)

    forecast=model.predict(n_periods=len(site))



    forecast=pd.DataFrame(forecast,index=site.index,columns=['Prediction'])
    
    
    #plot the predictions for validation set
    plt.plot(forecast,color='orange')
    plt.plot(site,color='black')
    plt.xlabel(u'时间')
    plt.ylabel(u'立案量')
    plt.legend(labels = [u'预测数据',u'实际数据'],loc=1)
    subplot.set_title(site_names[i])
    plt.tight_layout()

    '''
    #calculate rmse
    rms = sqrt(mean_squared_error(valid,forecast))
    print(rms)
    '''
    layout_num=layout_num+1








plt.show()



    
'''
#divide into train and validation set
train = data[:int(0.7*(len(data)))]
valid = data[int(0.7*(len(data))):]
 
#preprocessing (since arima takes univariate series as input)
train.drop('Month',axis=1,inplace=True)
valid.drop('Month',axis=1,inplace=True)
 


#plotting the data
train['International airline passengers'].plot()
valid['International airline passengers'].plot()
'''




 







