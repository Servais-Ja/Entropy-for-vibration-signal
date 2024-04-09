import numpy as np
import math
from scipy.stats import norm
from scipy.integrate import quad
from scipy.signal import argrelextrema


########################################################################################################################
# 多种熵的计算函数
########################################################################################################################
# 斜率熵计算函数，输入序列seq，及参数m、gamma、delta即可得该序列对应的熵，文献中m=3，gamma=1，delta=0.001
# m为嵌入维数，gamma和delta分别为划分数据的高低两个间断点
# detail用于精细复合多尺度熵的计算，为True时不输出熵值，而是输出模式及其出现频次的字典
# 感觉实际上也可以称作基于波动的香农熵（功率谱熵）？
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
        #p = k[1] / (N-m+1)
        slopen=slopen-p*math.log(p,2)
    if detail:
        patterndic={}
        for k in allpattern:
            patterndic[str(k[0])]=k[1]
        return patterndic
    return slopen

# ！！！计算复杂度太高，计算时间太长
# 散布熵计算函数，输入序列seq，及参数m、c、d即可得该序列对应的熵
# m为嵌入维数，c为类别个数，d为时延
def fun_pdf(x):
    return norm.pdf(x, a, b)
def Disen(seq, m=3, c=3, d=1, detail=False):
    N=len(seq)
    seq=np.array(seq)
    avg=np.mean(seq)
    std=np.std(seq)
    seq=np.array([quad(lambda x:norm.pdf(x, avg, std), -np.inf, y) for y in seq])
    seq=np.rint(c*seq+0.5)
    patterndic={}
    for i in range(N-(m-1)*d):
        pattern=str(seq[i:i+(m-1)*d+1:d])
        if pattern in patterndic.keys():
            patterndic[pattern]=patterndic[pattern]+1
        else:
            patterndic[pattern]=1
    length=sum(patterndic.values())
    pseq = [patterndic[i] / length * math.log(patterndic[i] / length, 2) for i in patterndic.keys()]
    if detail:
        return patterndic
    return -sum(pseq)

# 香农熵计算函数，输入序列seq，及参数n即可得该序列对应的熵，本函数默认参数根据在本数据集上的经验得到
# n为类别个数
# 香农熵的可采取的归一化方法上包括minmax，sigmoid，tan，本程序只实现了minmax
# 功率谱熵Power spectral entropy
def Shen(seq, n=10, detail=False):
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

# 注意熵计算函数，输入序列seq，及参数threshold即可得该序列对应的熵，本函数默认参数根据在本数据集上的经验得到
# 相临极大值极小值差需大于threshold才能被视为有效的极大值极小值
# 函数get_extremes用于计算交替的极大值和极小值
def get_extremes(seq, threshold, window):
    output_sequence = []
    output_index = []
    is_max = seq[0] < seq[1]
    initial = is_max
    threshold_value=threshold*np.std(np.array(seq))
    if window:
        for i in range(1, len(seq) - 1):
            if (is_max and seq[i] > seq[i - 1] and seq[i] > seq[i + 1]) or (not is_max and seq[i] < seq[i - 1] and seq[i] < seq[i + 1]):
                if len(output_sequence) == 0 or abs(seq[i] - output_sequence[-1]) > threshold*np.std(np.array(seq[i-10:i+10])):
                    output_sequence.append(seq[i])
                    output_index.append(i)
                    is_max = not is_max
    else:
        for i in range(1, len(seq) - 1):
            if (is_max and seq[i] > seq[i - 1] and seq[i] > seq[i + 1]) or (not is_max and seq[i] < seq[i - 1] and seq[i] < seq[i + 1]):
                if len(output_sequence) == 0 or abs(seq[i] - output_sequence[-1]) > threshold_value:
                    output_sequence.append(seq[i])
                    output_index.append(i)
                    is_max = not is_max
    return output_index, initial

def Aten(seq, threshold=0.1, window=False, detail=False):
    index, initial = get_extremes(seq, threshold, window)
    is_max=initial
    max_index = []
    min_index = []
    #seq=np.array(seq) seq(index)
    for i in index:
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

# ！！！计算复杂度太高，计算时间太长
# 近似熵计算函数，输入序列seq，及参数m、r即可得该序列对应的熵，文献中m=2， r=0.2*序列标准差（0.1~0.25）
# m为嵌入维数，r为相似容限与序列标准差的比值
def distance(seq1, seq2):
    return max(abs(np.array(seq1)-np.array(seq2)))

def Apen(seq, m=3, r=0.2):
    r = r*np.std(np.array(seq))
    N = len(seq)
    value1=0
    value2=0
    for k in range(2):
        seq_list = []
        m = m+k
        for i in range(N-m+1):
            seq_list.append(seq[i:i+m])
        lst = []
        for i in seq_list:
            distance_list=[]
            for j in seq_list:
                distance_list.append(distance(i, j))
            lst.append(np.sum(np.array(distance_list) < r) / (N - m + 1))
        if k == 0:
            value1 = sum([math.log(i) for i in lst])/N-m+1
        else:
            value2 = sum([math.log(i) for i in lst])/N-m+1
    return value1-value2

# 样本熵计算函数，输入序列seq，及参数m、r即可得该序列对应的熵，文献中m=2， r=0.2*序列标准差（0.1~0.25）
# m为嵌入维数，r为相似容限与序列标准差的比值
def Smpen(seq, m=3, r=0.2):
    r = r*np.std(np.array(seq))
    N = len(seq)
    value1=0
    value2=0
    for k in range(2):
        seq_list = []
        m = m+k
        for i in range(N-m+1):
            seq_list.append(seq[i:i+m])
        lst = []
        for i in seq_list:
            distance_list=[]
            for j in seq_list:
                distance_list.append(distance(i, j))
            lst.append((np.sum(np.array(distance_list) < r) - 1) / (N - m))
        if k == 0:
            value1 = sum(lst) / N - m + 1
        else:
            value2 = sum(lst) / N - m + 1
    return -math.log(value2/value1)

# 模糊熵计算函数，输入序列seq，及参数m、r、n即可得该序列对应的熵，文献中m=2， r=0.2*序列标准差（0.1~0.25），n=2
# m为嵌入维数，r为相似容限与序列标准差的比值
def Fuzen(seq, m=3, r=0.2, n=2):
    r = r*np.std(np.array(seq))
    N = len(seq)
    value1=0
    value2=0
    for k in range(2):
        seq_list = []
        m = m+k
        for i in range(N-m+1):
            seq_list_arr=np.array(seq[i:i+m])-sum(seq[i:i+m])/m
            seq_list.append(seq_list_arr.tolist())
        lst = []
        for i in seq_list:
            distance_list=[]
            for j in seq_list:
                distance_list.append(distance(i, j))
            lst.append((sum([math.exp(-(i**n/r)) for i in distance_list])-1) / (N - m))
        if k == 0:
            value1 = sum(lst) / N - m + 1
        else:
            value2 = sum(lst) / N - m + 1
    return -math.log(value2/value1)

# 排列熵计算函数，输入序列seq，及参数m、d即可得该序列对应的熵
# m为嵌入维数，d为时延
def Pren(seq, m=3, d=1, detail=False):
    patterndic={}
    N=len(seq)
    for i in range(N-(m-1)*d):
        sub_seq = seq[i:i+(m-1)*d+1:d]
        pattern = str(np.argsort(np.array(sub_seq)))
        if pattern in patterndic.keys():
            patterndic[pattern] = patterndic[pattern]+1
        else:
            patterndic[pattern] = 1
    length=sum(patterndic.values())
    if detail:
        return patterndic
    return sum([-patterndic[i]/length*math.log(patterndic[i]/length) for i in patterndic.keys()])

# 冒泡熵（Bubble Entropy）###0
# 分布熵（Distribution Entropy）###4
# 余弦相似熵（改进的近似熵）
# 相位熵 #
# 模糊散布熵 ###3


########################################################################################################################
# 改进的斜率熵计算函数
########################################################################################################################
def improved_Slopen(seq, en=Shen, detail=False, **kwargs):#seq是列表或np数组都可以
    arr = np.array(seq)[1:] - np.array(seq)[:-1]
    seq_difference = arr.tolist()
    return en(seq_difference, **kwargs, detail=detail)


########################################################################################################################
# 多种粗粒化方法
########################################################################################################################
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

# 粗粒化函数，输入序列seq，粗粒化参数尺度k，以及粗粒化方法（默认为求平均值）即可得相应的粗粒化后的序列
def ms(seq,k,func=average_coarse):
    n=len(seq)//k
    seq_=[]
    for i in range(n):
        seq_.append(func(seq[i*k:(i+1)*k]))
    return seq_


########################################################################################################################
# 多种熵的改进算法
########################################################################################################################
# 多尺度熵
def Multiscale(seq,tao,entropy,**kwargs):
    Mseq=[]
    for k in range(1,tao+1):
        Mseq_=ms(seq,k)
        Mseq.append(entropy(Mseq_,**kwargs))
    return Mseq

# 时移多尺度熵
def Time_shift_multiscale(seq,tao,entropy,**kwargs):#注意按相应熵函数的参数名输入#注意时移多尺度的第一个即是原序列
    TSMseq=[]
    for k in range(1,tao+1):
        value=0
        for i in range(k):
            value=value+entropy(seq[i::k],**kwargs)
        TSMen=value/k
        TSMseq.append(TSMen)
    return TSMseq

# 复合多尺度熵
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

# 精细复合多尺度熵(注意需增加detail=True)
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





def value_test():
    list4test=[8.2, 8.1, 4.4, 3.6, 5.3, 5.4, 8.3, 1.9, 3.7, 8.6, 9.6, 9, 6, 8.7, 6.7, 3.3, 2, 2.5, 2.7, 4.6, 9.1, 1, 3.1, 1.7, 4.1, 3.8, 6.4, 1.3, 5.7, 3.4, 2.4, 2.1, 4.2]
    print('Result:')
    print(Slopen(list4test, 3, 1, 0.001))
    print(Multiscale(list4test, 3, Slopen, m=3, gamma=1, delta=0.001))

if __name__=='__main__':
    value_test()