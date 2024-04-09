clear all 
clc

SearchAgents_no=30; % Number of search agents 种群数量
Function_name='F8'; % 1-7  单峰测试函数 8-13 多峰测试函数 14-23 固定维多峰测试函数
Max_iteration=500; % Maximum numbef of iterations 设定最大迭代次数
[lb,ub,dim,fobj]=Get_Functions_details(Function_name);  %设定边界以及优化函数
%Load details of the selected benchmark function
[Best_pos11,Best_score11,PSO_curve]=PSO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj); %开始优化
[Best_score22,Best_pos22,GWO_cg_curve]=GWO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
[Best_score33,Best_pos33,SCA_cg_curve]=SCA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
[Best_pos44,Best_score44,PIO_cg_curve]=PIO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
[Best_pos55,Best_score55,IPIO_cg_curve]=IPIO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%% 结果对比
figure('Position',[269   240   660   290])
%Draw search space
subplot(1,2,1);
func_plot(Function_name);
title('Parameter space')
xlabel('x_1');
ylabel('x_2');
zlabel([Function_name,'( x_1 , x_2 )'])

%Draw objective space
subplot(1,2,2);
semilogy(PSO_curve,'Color','b','linewidth',1.5)
hold on
semilogy(GWO_cg_curve,'Color','g','linewidth',1.5)
hold on
semilogy(SCA_cg_curve,'Color','k','linewidth',1.5)
hold on
semilogy(PIO_cg_curve,'Color','r','linewidth',1.5)
hold on
semilogy(IPIO_cg_curve,'Color','m','linewidth',1.5)
hold on
title('Objective space')
xlabel('Iteration');
ylabel('Best score obtained so far');

axis tight
grid on
box on
legend('PSO','GWO','SCA','PIO','IPIO')

display(['The best solution obtained by PSO is : ', num2str(Best_pos11)]);
display(['The best optimal value of the objective funciton found by PSO is : ', num2str(Best_score11)]);

display(['The best solution obtained by GWO is : ', num2str(Best_pos22)]);
display(['The best optimal value of the objective funciton found by GWO is : ', num2str(Best_score22)]);

display(['The best solution obtained by SCA is : ', num2str(Best_pos33)]);
display(['The best optimal value of the objective funciton found by SCA is : ', num2str(Best_score33)]);

display(['The best solution obtained by PIO is : ', num2str(Best_pos44)]);
display(['The best optimal value of the objective funciton found by PIO is : ', num2str(Best_score44)]);

display(['The best solution obtained by IPIO is : ', num2str(Best_pos55)]);
display(['The best optimal value of the objective funciton found by IPIO is : ', num2str(Best_score55)]);
