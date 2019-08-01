# -
本项目用到时间序列分析与预测、深度学习、数据挖掘的相关知识

由于不知道数据是否可以公开，所以仓库设置成私密了

注意事项：
1.不要改动备份数据的内容，【code】文件夹中提供了测试数据，可以在【code】文件夹中添加测试数据
2.将生成的图表放入【图表】文件夹中对应的位置，方便写论文时使用
3.将值得说明的文字内容以及资料放入【论文初稿】文件夹的对应位置，方便写论文时使用

=====================================================================================================================================================================

2019.7.17

现在的任务：
1. 现有LSTM算法和ARMA算法精度计算、误差计算、出图完善 
2.实现刚刚的2-3个其他算法 
3.还要分析一下站点数据之间的相关性 
4.对LSTM设置不同 time lag(delay) 和每层神经元节点数产生的误差进行实验和比较 
5.选取效果最好的time lag和每层神经元节点数，然后跑筛选后的数据，五种类型分开跑，
	然后按照城市系统工程研究中心老师的要求，分析什么样子的数据预测效果好，这样子的数据对应于什么街道，什么案件 
6.依据上述各实验比对结果，以及最后的分析进行论文撰写。

=====================================================================================================================================================================



=====================================================================================================================================================================

2019.7.19

改进了ARMA，积存渣土的西长安街街道的数据跑不通
跑通的都放到了图标里了
还有待改进

下午ARMA已经完成

整体情况是
乱堆物料整体最准，非法小广告整体最不准 积存渣土的数据模型不接受
至于预测效果的准确程度和什么有关，还希望能进一步仔细观察对比图表数据
初步观察是和数据平稳度与数据的数量级有关
打算开始ARIMA，增强原有数据平稳度

=====================================================================================================================================================================

2019.7.22

完成SVR，预测效果依然不理想
差分法成功，数据平稳化成功
关于ARMA和ARIMA，有的站点跑不同已经排除数据不平稳的影响，推测是由于p，q定阶不准造成的，接下来会修改定阶的办法

Auto-Arima可以自动修改p和q，但预测出的数据仅仅是一个趋势。

样本数据没有特定的趋势或季节性，因此推测LSTM依然是最合适的办法

接下来会先尝试新的定阶的方法，然后便要开始着手调试LSTM了。


=====================================================================================================================================================================

2019.7.30

修改LSTM 最近还在了解深度学习算法以及Keras的应用，这块是难点
现在LSTM可以跑全部站点
接下来主要是准确度方面的研究
首先是要分站点归一化，这个还要和学长讨论一下
其次是控制变量实验，比如改变学习率，改变层数之类的，看哪个准确度最高。


=====================================================================================================================================================================

2019.8.1
相关性分析完成，各站点相关性不大，看来还是得分站点测试，根据这个改代码是最难的部分
等改完代码就可以测试了（如果能改完的话。。）
测试规划：
epoch=10,40,60,100分析结果
节点数=600，800，1000分析结果
delay=4,8,12分析结果
batch_size=12，24，48分析结果
学习率这个比较复杂，还得再想想

=====================================================================================================================================================================