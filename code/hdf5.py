import h5py  #导入工具包  
import numpy as np

#HDF5的读取：  
f = h5py.File('$8_weights.hdf5','r')   #打开h5文件  
a=f.keys()                            #可以查看所有的主键  ：在这里是：【data】,[label]
#a = f['lstm_1'][:]                    #取出主键为data的所有的键值  
print(a)

f.close() 