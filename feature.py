#安装所需的库
import numpy as np
import slopen
import scipy.io as sio
n=2000
#文件读取(列表或np.array均可)
#多个文件一块儿读取，记得变量名要变
#读入文件
dicmat1=sio.loadmat('./data/load_current_10A.mat')###
#读取文件内数据，注意mat文件中变量名不完全相同
arr1=[]
arr1.append(np.array(dicmat1["Data1_AI_0"]).flatten())###
arr1.append(np.array(dicmat1["Data1_AI_1"]).flatten())
arr1.append(np.array(dicmat1["Data1_AI_2"]).flatten())
arr1.append(np.array(dicmat1["Data1_AI_3"]).flatten())
arr1.append(np.array(dicmat1["Data1_AI_5"]).flatten())
arr1.append(np.array(dicmat1["Data1_AI_7"]).flatten())
#数据划分
short_circuit=[]
for i in range(6):
    k=len(arr1[i])//n
    for j in range(k):
        short_circuit.append(arr1[i][j*n:(j+1)*n].tolist())

#特征提取
#不能取太大，不然出现不了重复的模式(3或4吧)
short_circuit_M=[]
short_circuit_TSM=[]
short_circuit_CM=[]
short_circuit_RCM=[]
for seq in short_circuit:
    #feature_seq_M=slopen.Multiscale(seq, 8, slopen.Slopen, m=3, gamma=1, delta=0.001)
    feature_seq_M=slopen.Multiscale(seq, 8, slopen.Disen)###
    feature_seq_TSM=slopen.Time_shift_multiscale(seq, 8, slopen.Disen)
    feature_seq_CM=slopen.Composite_multiscale(seq, 8, slopen.Disen)
    feature_seq_RCM = slopen.Refined_composite_multiscale(seq, 8, slopen.Disen)
    short_circuit_M.append(feature_seq_M)
    short_circuit_TSM.append(feature_seq_TSM)
    short_circuit_CM.append(feature_seq_CM)
    short_circuit_RCM.append(feature_seq_RCM)
#将提取的特征存入文件
short_circuit_M_arr=np.unique(np.array(short_circuit_M),axis=0)
short_circuit_TSM_arr=np.unique(np.array(short_circuit_TSM),axis=0)
short_circuit_CM_arr=np.unique(np.array(short_circuit_CM),axis=0)
#short_circuit_RCM_arr=np.unique(np.array(short_circuit_RCM),axis=0)
np.savetxt('./feature/short_circuit_M.txt',short_circuit_M_arr,fmt='%.5f',delimiter='\t')
np.savetxt('./feature/short_circuit_TSM.txt',short_circuit_TSM_arr,fmt='%.5f',delimiter='\t')
np.savetxt('./feature/short_circuit_CM.txt',short_circuit_CM_arr,fmt='%.5f',delimiter='\t')
#np.savetxt('./feature/short_circuit_RCM.txt',short_circuit_RCM_arr,fmt='%.5f',delimiter='\t')