#用于将.csv转化为相关性分析所需要的文件格式
import pandas as pd
import numpy as np

FILE_NAME = u"无照经营"
df=pd.read_csv(FILE_NAME+"-所有街道数据.csv",encoding='gbk')

df_new=pd.DataFrame()


ROW = df.shape[0] #数据行数=表格行数-1（减表头）
DATA_SIZE=48 #数据量 每个街道有DATA_SIZE个月的数据
SITE_SIZE=int(ROW/DATA_SIZE)#站点个数


hourlyData=df.values[:ROW,3]

#将数据按站点分为SITE_SIZE组
site_data=[]  #站点数据列表
site_cnames=[] #站点名字列表
for num in range(0,SITE_SIZE):
    site_cnames.append(df.at[num*DATA_SIZE, u'事发街道'])
    if num==0:
        site_data.append(hourlyData[0:DATA_SIZE])
    else:
        temp=hourlyData[num*DATA_SIZE:(num+1)*DATA_SIZE]
        site_data.append(temp)


temp = np.array(site_data) 

site_data = np.array(site_data) 
site_data.reshape(DATA_SIZE,SITE_SIZE)
site_data=site_data.T

site_cnames=np.array(site_cnames) 

result=np.vstack((site_cnames,site_data)) #合并两个矩阵

data = pd.DataFrame(result)

df_new=df_new.append(data)

df_new.to_csv("./"+FILE_NAME+"-各站点相关性分析.csv",index=False,encoding='gbk')

