function [outputArg] = SCN_fun(parameterlist)
%SCN_FUN 此处显示有关此函数的摘要
%   此处显示详细说明
L_max=round(parameterlist(1));
T_max=round(parameterlist(2));
tol=parameterlist(3);
Data{1}=table2array(readtable('D:\DownLoad\WeChat Files\wxid_llkq0xb29uu622\FileStorage\File\2024-03\feature\feature\guoli_TSM.txt'));
Data{2}=table2array(readtable('D:\DownLoad\WeChat Files\wxid_llkq0xb29uu622\FileStorage\File\2024-03\feature\feature\over_circuit_TSM.txt'));
Data{3}=table2array(readtable('D:\DownLoad\WeChat Files\wxid_llkq0xb29uu622\FileStorage\File\2024-03\feature\feature\qianli_TSM.txt'));
Data{4}=table2array(readtable('D:\DownLoad\WeChat Files\wxid_llkq0xb29uu622\FileStorage\File\2024-03\feature\feature\short_circuit_TSM.txt'));
Data{5}=table2array(readtable('D:\DownLoad\WeChat Files\wxid_llkq0xb29uu622\FileStorage\File\2024-03\feature\feature\under_circuit_TSM.txt'));
Data{6}=table2array(readtable('D:\DownLoad\WeChat Files\wxid_llkq0xb29uu622\FileStorage\File\2024-03\feature\feature\zhengli_TSM.txt'));
X=cat(1,Data{:});
Y=cat(1,ones(length(Data{1}),1),2*ones(length(Data{2}),1),3*ones(length(Data{3}),1),4*ones(length(Data{4}),1),5*ones(length(Data{5}),1),6*ones(length(Data{6}),1));
Data=[Y X];                                             %经处理后得到的数据，第一列为标签，之后为特征

num_class = length(unique(Data(:, 1)));                 %类别数（Excel第一列放类别）

Indices=[];
for i=1:num_class
    mid_res = Data((Data(:, 1) == i), :);                       % 取出类别i的样本
    mid_size = size(mid_res, 1);                                % 类别i样本个数
    indices=crossvalind('Kfold',mid_size,5);
    Indices=cat(1,Indices,indices);
end
for jj=1:5
    %% 划分训练集和测试集
    test=(Indices==jj);
    train=~test;
    Data_test=Data(test,:);
    Data_train=Data(train,:);
    P_train=Data_train(1: length(Data_train), 2: end );
    P_test=Data_test(1: length(Data_test), 2: end );
    T_train=Data_train(1: length(Data_train), 1);
    T_test=Data_test(1: length(Data_test), 1);
    [train_x,train_ps]=mapminmax(P_train',0,1);                 % 归一化到0,1之间，其中train_ps保留归一化函数的详细信息
    test_x=mapminmax('apply',P_test',train_ps);                 % 相同方式进行测试集归一化

    %% 标签转换为onehot
    train_Y=zeros(size(T_train,1),2);                               % ???独热编码的话应该是4而不是2吧，不过影响不大就是了
    for i =1:size(T_train,1)
        train_Y(i,T_train(i))=1;
    end
    test_Y=zeros(size(T_test,1),2);
    for i =1:size(T_test,1)
        test_Y(i,T_test(i))=1;
    end

    P_train = double(train_x)' ;                    % 转换为double类型并转置
    P_test  = double(test_x)' ;
    T_train = double(train_Y);
    T_test  = double(test_Y);
    
    %% Parameter Setting SCN参数设置
    Lamdas=[0.5, 1, 5, 10, 30, 50, 100, 150, 200, 250];
    r=[0.7,0.8,0.9, 0.99, 0.9999, 0.99999, 0.999999];
    %% Model Initialization 选择模型SCN
    M = SCN(L_max, T_max, tol, Lamdas, r);
    %% Model Training 训练SCN模型
    [M, ~] = M.Classification(P_train, T_train, 0);

    O2 = M.GetLabel(P_test);

    for i =1:size(O2 ,1)                            % ???应该是测试集预测值
        [~,index3]=find(O2(i,:)==1);
        c(i)=index3;  
    end
    for i =1:size(T_test ,1)                        % 测试集实际标签
        [~,index4]=find(T_test(i,:)==1);
        d(i)=index4;  
    end
    testlabel_index_actuala=categorical(d);         % 对真实值和预测值进行categorical变换
    testlabel_index_expecteda=categorical(c);
    [A,~] = confusionmat(testlabel_index_actuala,testlabel_index_expecteda);%得到混淆矩阵，其中~表示忽略输出参数
    Accuracy(jj)=trace(A)/sum(sum(A));
end
outputArg=mean(Accuracy);
end

