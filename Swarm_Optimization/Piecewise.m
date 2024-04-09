%% Piecewise映射（仅在initializationNew中使用）
function [x] = Piecewise(Max_iter)
x(1)=rand; %初始点
P=0.4;
for i=1:Max_iter-1
    if x(i)>=0 && x(i)<P
        x(i+1)=x(i)/P;
    end
    if x(i)>=P && x(i)<0.5
        x(i+1)=(x(i)-P)/(0.5-P);
    end
    if x(i)>=0.5 && x(i)<1-P
        x(i+1)=(1-P-x(i))/(0.5-P);
    end
    if x(i)>=1-P && x(i)<1
        x(i+1)=(1-x(i))/P;
    end    
end
 end