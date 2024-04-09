%_________________________________________________________________________%
% 鸽群优化算法            %
%_________________________________________________________________________%
function [Best_pos,Best_score,curve]=IPIO(pop,Max_iter,lb,ub,dim,fobj)
Nc1= round(Max_iter*0.7);%地图因子
Nc2= Max_iter - Nc1;  %指南因子
JY=1;
JYNumber = round(JY*pop);%精英麻雀个数
if(max(size(ub)) == 1)
   ub = ub.*ones(1,dim);
   lb = lb.*ones(1,dim);  
end

%种群初始化
X0=initializationNew(pop,dim,ub,lb);
X = X0;
%计算初始适应度值
fitness = zeros(1,pop);
for i = 1:pop
   fitness(i) =  fobj(X(i,:));
end
 [fitness, index]= sort(fitness);%排序
GBestF = fitness(1);%全局最优适应度值
for i = 1:pop
    X(i,:) = X0(index(i),:);
end
curve=zeros(1,Max_iter);
GBestX = X(1,:);%全局最优位置
X_new = X;
%初始速度
Vec = rand(pop,dim);
VecNew = Vec;
%地图更新
for t = 1: Nc1
    Vec = VecNew;
    for i = 1:pop
        R = rand;
       %速度更新
       TempV=  Vec(i,:) + rand.*(GBestX - X(i,:));
       %位置更新
       TempPosition = X(i,:).*(1-exp(-R*t)) + TempV;
        %边界检查
        for j = 1:dim
           if(TempPosition(j)<lb(j) || TempPosition(j)>ub(j)) 
               TempPosition(j) = lb(j) + rand.*(ub(j) - lb(j));
               TempV(j) = rand;
           end
        end
       X_new(i,:) = TempPosition;   
       VecNew(i,:)=TempV;
    end      
   for j=1:pop
    fitness_new(j) = fobj(X_new(j,:));
   end
   X = X_new;
   fitness = fitness_new;
    %排序更新
   [fitness, index]= sort(fitness);%排序
   for j = 1:pop
      X(j,:) = X(index(j),:);
   end
   if(fitness(1) < GBestF)
       GBestF = fitness(1);
        GBestX = X(1,:);   
   end
   %% 翻筋斗机制
for j = 1:JYNumber
    Y(j,:)=X(j,:)+2*(rand*GBestX-rand*X(j,:));
      for a = 1: dim
           if(Y(j,a)>ub)
              Y(j,a) =ub(a)-rand*(ub(a)-lb(a));
           end
           if(Y(j,a)<lb)
              Y(j,a) =lb(a)+rand*(ub(a)-lb(a));
           end
      end 
end %这个end要注意
   %% 后续处理方式
for j=1:JYNumber+pop
          if(j<pop+1)
          EE(j,:)=X(j,:);
          fit(j)=fitness(j);
          else
          EE(j,:)=Y(j-pop,:);  
          fit(j) = fobj(Y(j-pop,:));
          end
  end 
 [NEWfitness, index]= sort(fit);%排序
   for j = 1:pop
      X(j,:) = EE(index(j),:);
   end

   GBestX=X(1,:);
   curve(t) = GBestF;
   
end
%指南更新
for t = 1:Nc2
   % 根据地标舍去后50%，并计算中心
   S = 0;
   for j = 1:round(pop/2)
      S = S + X(j,:).*fitness(j);      
   end

   Xcenter = S./((pop/2)*sum(fitness(1:round(pop/2))));
   %根据中心更新种群
   for i = 1:round(pop/2)
       for j = 1:dim
            Temp = X(i,j) + rand.*(Xcenter(j) - X(i,j));
            while Temp<lb(j) || Temp>ub(j)
                Temp = X(i,j) + rand.*(Xcenter(j) - X(i,j));
            end
       end
       X(i,:) = Temp;     
   end         
   for j=1:pop
    fitness(j) = fobj(X(j,:));
   end
    %排序更新
   [fitness, index]= sort(fitness);%排序
   for j = 1:pop
      X(j,:) = X(index(j),:);
   end
    if(fitness(1) < GBestF)
       GBestF = fitness(1);
        GBestX = X(1,:);   
    end
    %% 翻筋斗机制
   for j = 1:JYNumber
    Y(j,:)=X(j,:)+2*(rand*GBestX-rand*X(j,:));
      for a = 1: dim
           if(Y(j,a)>ub)
              Y(j,a) =ub(a)-rand*(ub(a)-lb(a));
           end
           if(Y(j,a)<lb)
              Y(j,a) =lb(a)+rand*(ub(a)-lb(a));
           end
      end 
end %这个end要注意
 
   %% 后续处理方式
  for j=1:JYNumber+pop
          if(j<pop+1)
          EE(j,:)=X(j,:);
          fit(j)=fitness(j);
          else
          EE(j,:)=Y(j-pop,:);  
          fit(j) = fobj(Y(j-pop,:));
          end
  end 
 [NEWfitness, index]= sort(fit);%排序
   for j = 1:pop
      X(j,:) = EE(index(j),:);
   end

   GBestX=X(1,:);
   curve(Nc1 + t) = GBestF;
end
Best_pos =GBestX;
Best_score = curve(end);
end