clear all 
clc
SearchAgents_no=10; % Number of search agents 种群数量

Max_iteration=20; % Maximum numbef of iterations 设定最大迭代次数

lb=[10,10,0.0001];
ub=[50,200,0.1];
dim=3;
fobj=@SCN_fun;
%Load details of the selected benchmark function
[Best_pos11,Best_score11,PSO_curve]=PSO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj); %开始优化
[Best_score22,Best_pos22,GWO_cg_curve]=GWO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
[Best_score33,Best_pos33,SCA_cg_curve]=SCA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%[Best_pos44,Best_score44,PIO_cg_curve]=PIO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%[Best_pos55,Best_score55,IPIO_cg_curve]=IPIO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%% 结果对比
PSO=Best_score11;
GWO=Best_score22;
SCA=Best_score33;
%PIO(i)=Best_score44;
%IPIO(i)=Best_score55;

display(['Accuracy using PSO is : ', num2str(Best_score11)]);
display(['Position using PSO is : ', num2str(Best_pos11)]);
display(['Accuracy using GWO is : ', num2str(Best_score22)]);
display(['Position using GWO is : ', num2str(Best_pos22)]);
display(['Accuracy using SCA is : ', num2str(Best_score33)]);
display(['Position using SCA is : ', num2str(Best_pos33)]);
%display(['PIO is : ', num2str(PIO)]);

%display(['IPIO is : ', num2str(IPIO)]);