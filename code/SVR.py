from __future__ import division
import pandas as pd
import time
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from matplotlib.font_manager import _rebuild
from matplotlib.pylab import mpl
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

rng = np.random.RandomState(0)

FILE_NAME=u"暴露垃圾-所有街道数据.csv"
df=pd.read_csv(FILE_NAME,encoding="gbk")

ROW = df.shape[0] #数据行数=表格行数-1（减表头）
COLUMN = 9 #数据列数=表格列数
DATA_SIZE= 48 #数据量 每个街道有DATA_SIZE个月的数据
inData=df.values[:ROW,3]
inData=inData.astype('float32')

SITE_SIZE=int(ROW/DATA_SIZE)

#设置绘图时的中文显示（需安装黑体字体）
_rebuild()
mpl.rcParams['font.sans-serif']=[u'SimHei']
mpl.rcParams['axes.unicode_minus']=False
#############################################################################
#将数据按站点分为SITE_SIZE组
site_names=[]  #站点数据列表
site_cnames=[] #站点名字列表
#在进行运算之前可以对数据进行归一化，进而降低loss
#scaler = MinMaxScaler()
#inData = scaler.fit_transform(inData.reshape(-1,1))

for num in range(0,SITE_SIZE):
    site_cnames.append(df.at[num*DATA_SIZE, u'事发街道'])
    if num==0:
        site_names.append(inData[0:DATA_SIZE])
    else:
        site_names.append(inData[num*DATA_SIZE:(num+1)*DATA_SIZE])
# 生成数据
layout_num = 0
plt.figure(figsize=(16,9))
for i in range(SITE_SIZE):
    X = np.linspace(1,48,48).reshape(-1,1)

    #X = 48 * rng.rand(10000, 1)
    y = site_names[i].ravel().T
    #y = np.sin(X).ravel()

    X_plot = np.linspace(1, 48, 48)[:, None]

#############################################################################
# 训练SVR模型

# 训练规模
    train_size = 100
# 初始化SVR
    svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})
# 记录训练时间
    t0 = time.time()
# 训练
    svr.fit(X[:train_size], y[:train_size])
    svr_fit = time.time() - t0

    t0 = time.time()
# 测试
    y_svr = svr.predict(X_plot)
    svr_predict = time.time() - t0


#############################################################################
# 对结果进行显示
    #y = scaler.inverse_transform(y.reshape(-1,1))
    #y_svr = scaler.inverse_transform(y_svr.reshape(-1,1))
    if (layout_num == 6):
        layout_num = 0
        plt.figure(figsize=(16, 9))
        plt.suptitle(u'分站点预测/实际值对比')

    subplot = plt.subplot(3, 2, layout_num + 1)
    plt.scatter(X[:train_size], y[:train_size], c='k', label='data', zorder=1)
    #plt.plot(y,c='b',label = 'real data')
    plt.plot( X_plot, y_svr, c='r',
         label='SVR predict data (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('SVR versus Kernel Ridge')
    plt.legend()
    plt.tight_layout()
    layout_num = layout_num + 1
plt.show()