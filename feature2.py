#安装所需的库
import numpy as np
import entropy
import scipy.io as sio
import os
n=2000
#文件读取(列表或np.array均可)
filelist=os.listdir('./Data/data2')
#读入文件
Data = []
for i in filelist:
    dicmat = sio.loadmat('./Data/data2/'+i)
    arr = dicmat["data"]
    for j in range(10):
        try:
            Data[j] = np.vstack((Data[j],arr[j*50:(j+1)*50]))
        except:
            Data.append(arr[j*50:(j+1)*50])

#特征提取
cnt=0
for i in Data:
    cnt=cnt+1
    feature_M = []
    feature_TSM = []
    feature_CM = []
    # feature_RCM = []
    for seq in i:
        feature_seq_M = entropy.Multiscale(seq, 8, entropy.Slopen)
        feature_seq_TSM = entropy.Time_shift_multiscale(seq, 8, entropy.Slopen)
        feature_seq_CM = entropy.Composite_multiscale(seq, 8, entropy.Slopen)
        # feature_seq_RCM = entropy.Refined_composite_multiscale(seq, 8, entropy.Disen)
        feature_M.append(feature_seq_M)
        feature_TSM.append(feature_seq_TSM)
        feature_CM.append(feature_seq_CM)
        # feature_RCM.append(feature_seq_RCM)
    feature_M_arr = np.array(feature_M)
    feature_TSM_arr = np.array(feature_TSM)
    feature_CM_arr = np.array(feature_CM)
    # feature_RCM_arr= np.array(short_circuit_RCM)
    np.savetxt('./feature/feature_M_'+str(cnt)+'.txt', feature_M_arr, fmt='%.5f', delimiter='\t')
    np.savetxt('./feature/feature_TSM_'+str(cnt)+'.txt', feature_TSM_arr, fmt='%.5f', delimiter='\t')
    np.savetxt('./feature/feature_CM_'+str(cnt)+'.txt', feature_CM_arr, fmt='%.5f', delimiter='\t')
    # np.savetxt('./feature/short_circuit_RCM_'+str(1)+'.txt',short_circuit_RCM_arr,fmt='%.5f',delimiter='\t')