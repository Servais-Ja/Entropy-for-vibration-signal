import numpy as np
import math
from scipy.signal import argrelextrema
#斜率熵计算函数，输入序列seq，及参数m、gamma、delta即可得该序列对应的熵，文献中m=3，gamma=1，delta=0.001
def Slopen(seq, m=3, gamma=1, delta=0.001, detail=False):#seq是列表或np数组都可以
    N=len(seq)
    arr=np.array(seq)
    allpattern=[]
    slopen=0
    for i in range(N-m+1):
        arri=arr[i:i+m]
        arri=arri[1:]-arri[:-1]
        arri[(arri>=-delta)&(arri<=delta)]=0
        arri[(arri>delta)&(arri<=gamma)]=1
        arri[arri>gamma]=2
        arri[(arri>=-gamma)&(arri<-delta)]=-1
        arri[(arri<-gamma)]=-2
        pattern=arri
        bfound=False
        for j in allpattern:
            if (j[0]==pattern).all():
                j[1]=j[1]+1
                bfound=True
                break
        if not bfound:
            allpattern.append([pattern,1])
    for k in allpattern:
        p=k[1]/len(allpattern)
        slopen=slopen-p*math.log(p,2)
    if detail:
        patterndic={}
        for k in allpattern:
            patterndic[str(k[0])]=k[1]
        return patterndic
    return slopen

#香农熵计算函数，n=20
#归一化方法上包括minmax，sigmoid，tan，只实现了minmax
#功率谱熵Power spectral entropy
def Shen(seq, n=20, detail=False):
    seq=np.array(seq)
    min_limit = min(seq)
    max_limit = max(seq)
    k=(max_limit-min_limit)/n
    seq[(seq >= min_limit) & (seq <= min_limit+k)] = 0
    for i in range(1,n):
        seq[(seq>min_limit+i*k)&(seq<=min_limit+(i+1)*k)] = i
    patterndic={}
    for i in seq:
        if i in patterndic.keys():
            patterndic[i]=patterndic[i]+1
        else:
            patterndic[i]=1
    length=len(seq)
    pseq = [patterndic[i]/length*math.log(patterndic[i]/length, 2) for i in patterndic.keys()]
    if detail:
        return patterndic
    return -sum(pseq)


#注意熵计算函数
#计算交替的极大值和极小值
def get_extremes(seq, threshold):
    output_sequence = []
    output_index = []
    is_max = seq[0] < seq[1]
    initial = is_max
    for i in range(1, len(seq) - 1):
        if (is_max and seq[i] > seq[i - 1] and seq[i] > seq[i + 1]) or (not is_max and seq[i] < seq[i - 1] and seq[i] < seq[i + 1]):
            if len(output_sequence) == 0 or abs(seq[i] - output_sequence[-1]) > threshold:
                output_sequence.append(seq[i])
                output_index.append(i)
                is_max = not is_max
    return output_index, initial
def Aten(seq, threshold=0.0001, detail=False):
    index, initial = get_extremes(seq, threshold)
    is_max=initial
    max_index = []
    min_index = []
    seq=np.array(seq)
    for i in seq[index]:
        if is_max:
            max_index.append(i)
        else:
            min_index.append(i)
        is_max = not is_max
    max_index=np.array(max_index)
    min_index=np.array(min_index)
    max_max=max_index[1:]-max_index[:-1]
    min_min=min_index[1:]-min_index[:-1]
    length = min(len(max_index), len(min_index))
    length1 = min(len(max_index)-1, len(min_index))
    length2 = min(len(max_index), len(min_index)-1)
    if initial:
        max_min = min_index[0:length]-max_index[0:length]
        min_max = max_index[1:length1+1]-min_index[0:length1]
    else:
        max_min = min_index[1:length2+1] - max_index[0:length2]
        min_max = max_index[0:length] - min_index[0:length]
    aten=0
    for i in [max_max,min_min,max_min,min_max]:
        patterndic={}
        for j in i:
            if j in patterndic.keys():
                patterndic[j]=patterndic[j]+1
            else:
                patterndic[j]=1
        parr=np.array(list(patterndic.values()))/len(i)
        aten=aten+sum([i*math.log(i,2) for i in parr])
    if detail:
        return patterndic
    return aten/4
"""
#写的不太行
def Aten_(seq, m, gamma, delta, detail=False):#seq是列表或np数组都可以
    seq = np.array(seq)
    max_index = argrelextrema(seq, np.greater)[0]
    min_index = argrelextrema(seq, np.less)[0]
    #max-max
    j = max_index[0]
    allpattern_max_max = {}
    for i in max_index[1:]:
        pattern = i-j
        j = i
        if pattern in allpattern_max_max.keys():
            allpattern_max_max[pattern] = allpattern_max_max[pattern]+1
        else:
            allpattern_max_max[pattern] = 1
    #min-min
    j = min_index[0]
    allpattern_min_min = {}
    for i in min_index[1:]:
        pattern = i - j
        j = i
        if pattern in allpattern_min_min.keys():
            allpattern_min_min[pattern] = allpattern_min_min[pattern] + 1
        else:
            allpattern_min_min[pattern] = 1
    return 1
"""

#多种粗粒化方法
def average_coarse(seq):
    return sum(seq)/len(seq)
def max_coarse(seq):
    return max(seq)
def min_coarse(seq):
    return min(seq)
def root_mean_square_coarse(seq):
    return math.sqrt(sum([i**2 for i in seq])/len(seq))
def variance_coarse(seq):
    return np.var(np.array(seq))

#粗粒化函数，输入序列seq，粗粒化参数尺度k，以及粗粒化方法（默认为求平均值）即可得相应的粗粒化后的序列
def ms(seq,k,func=average_coarse):
    n=len(seq)//k
    seq_=[]
    for i in range(n):
        seq_.append(func(seq[i*k:(i+1)*k]))
    return seq_

#多尺度熵
def Multiscale(seq,tao,entropy,**kwargs):
    Mseq=[]
    for k in range(1,tao+1):
        Mseq_=ms(seq,k)
        Mseq.append(entropy(Mseq_,**kwargs))
    return Mseq

#时移多尺度熵
def Time_shift_multiscale(seq,tao,entropy,**kwargs):#注意按相应熵函数的参数名输入#注意时移多尺度的第一个即是原序列
    TSMseq=[]
    for k in range(1,tao+1):
        value=0
        for i in range(k):
            value=value+entropy(seq[i::k],**kwargs)
        TSMen=value/k
        TSMseq.append(TSMen)
    return TSMseq

#复合多尺度熵
def Composite_multiscale(seq,tao,entropy,**kwargs):
    CMseq=[]
    for k in range(1,tao+1):
        value=0
        for i in range(k):
            CMseq_=ms(seq[i:],k)
            value=value+entropy(CMseq_,**kwargs)
        CMen=value/k
        CMseq.append(CMen)
    return CMseq

#精细复合多尺度熵(注意需增加detail=True，只能用于slopen)
def Refined_composite_multiscale(seq,tao,entropy,**kwargs):
    RCMseq = []
    for k in range(1, tao + 1):
        RCMseq_=[]
        allpatterndic={}
        RCMen=0
        for i in range(k):
            RCMseq_=ms(seq[i:], k)
            patterndic = entropy(RCMseq_, **kwargs, detail=True)
            for j in patterndic.keys():
                if j in allpatterndic.keys():
                    allpatterndic[j]=allpatterndic[j]+patterndic[j]
                else:
                    allpatterndic[j]=patterndic[j]
        for k in allpatterndic.keys():
            p = allpatterndic[k] / len(allpatterndic.keys())
            RCMen = RCMen - p * math.log(p, 2)
        RCMseq.append(RCMen)
    return RCMseq

#层次多尺度熵

#多种粗粒化方法
def value_test():
    list4test=[8.2, 8.1, 4.4, 3.6, 5.3, 5.4, 8.3, 1.9, 3.7, 8.6, 9.6, 9, 6, 8.7, 6.7, 3.3, 2, 2.5, 2.7, 4.6, 9.1, 1, 3.1, 1.7, 4.1, 3.8, 6.4, 1.3, 5.7, 3.4, 2.4, 2.1, 4.2]
    print('Result:')
    print(Slopen(list4test, 3, 1, 0.001))
    print(Multiscale(list4test, 3, Slopen, m=3, gamma=1, delta=0.001))

if __name__=='__main__':
    value_test()