#安装所需的库
import numpy as np
import entropy
import scipy.io as sio
n=2048
#文件读取(列表或np.array均可)
#多个文件一块儿读取，记得变量名要变
#读入文件
dicmat=[]
folder_list=['./Data/data1/load_current_10A.mat', './Data/data1/NoLoad_340V.mat']# 文件夹列表
for folder in folder_list:
    dicmat.append(sio.loadmat(folder))###
#读取文件内数据，注意mat文件中变量名不完全相同

class_list=[]
for i in range(len(dicmat)):
    var_list = [j for j in dicmat[i].keys() if 'Data1_AI_' in j]
    arr1 = []
    for k in var_list:
        arr1.append(np.array(dicmat[i][k]).flatten())
    # 数据划分
    split_arr1 = []
    for l in range(len(arr1)):
        k = len(arr1[l])//n
        for j in range(k):
            split_arr1.append(arr1[l][j*n:(j+1)*n].tolist())
    class_list.append(split_arr1)

cnt=0
for seq_list in class_list:
    #特征提取
    #不能取太大，不然出现不了重复的模式(3或4吧)
    M=[]
    TSM=[]
    CM=[]
    #short_circuit_RCM=[]
    for seq in seq_list:
        #feature_seq_M=entropy.Multiscale(seq, 8, entropy.Slopen, m=3, gamma=1, delta=0.001)
        feature_seq_M=entropy.Multiscale(seq, 8, entropy.Aten)###
        feature_seq_TSM=entropy.Time_shift_multiscale(seq, 8, entropy.Aten)
        feature_seq_CM=entropy.Composite_multiscale(seq, 8, entropy.Aten)
        #feature_seq_RCM = entropy.Refined_composite_multiscale(seq, 8, entropy.Apen)
        M.append(feature_seq_M)
        TSM.append(feature_seq_TSM)
        CM.append(feature_seq_CM)
        #short_circuit_RCM.append(feature_seq_RCM)
    #将提取的特征存入文件

    M_arr=np.array(M)
    TSM_arr=np.array(TSM)
    CM_arr=np.array(CM)
    #short_circuit_RCM_arr=np.unique(np.array(short_circuit_RCM),axis=0)

    cnt += 1
    np.savetxt('./feature/class'+str(cnt)+'_M.txt',M_arr,fmt='%.5f',delimiter='\t')
    np.savetxt('./feature/class'+str(cnt)+'_TSM.txt',TSM_arr,fmt='%.5f',delimiter='\t')
    np.savetxt('./feature/class'+str(cnt)+'_CM.txt',CM_arr,fmt='%.5f',delimiter='\t')
    #np.savetxt('./feature/short_circuit_RCM.txt',short_circuit_RCM_arr,fmt='%.5f',delimiter='\t')
    print('class'+str(cnt)+' finished')