%_________________________________________________________________________%
% ��Ⱥ�Ż��㷨            %
%_________________________________________________________________________%
function [Best_pos,Best_score,curve]=PIO(pop,Max_iter,lb,ub,dim,fobj)
Nc1= round(Max_iter*0.7);%��ͼ����
Nc2= Max_iter - Nc1;  %ָ������
if(max(size(ub)) == 1)
   ub = ub.*ones(1,dim);
   lb = lb.*ones(1,dim);  
end

%��Ⱥ��ʼ��
X0=initialization(pop,dim,ub,lb);
X = X0;
%�����ʼ��Ӧ��ֵ
fitness = zeros(1,pop);
for i = 1:pop
   fitness(i) =  fobj(X(i,:));
end
 [fitness, index]= sort(fitness);%����
GBestF = fitness(1);%ȫ��������Ӧ��ֵ
for i = 1:pop
    X(i,:) = X0(index(i),:);
end
curve=zeros(1,Max_iter);
GBestX = X(1,:);%ȫ������λ��
X_new = X;
%��ʼ�ٶ�
Vec = rand(pop,dim);
VecNew = Vec;
%��ͼ����
for t = 1: Nc1
    Vec = VecNew;
    for i = 1:pop
        R = rand;
       %�ٶȸ���
       TempV=  Vec(i,:) + rand.*(GBestX - X(i,:));
       %λ�ø���
       TempPosition = X(i,:).*(1-exp(-R*t)) + TempV;
       %�߽���
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
    %�������
   [fitness, index]= sort(fitness);%����
   for j = 1:pop
      X(j,:) = X(index(j),:);
   end
   if(fitness(1) < GBestF)
       GBestF = fitness(1);
        GBestX = X(1,:);   
    end
   curve(t) = GBestF;
   
end
%ָ�ϸ���
for t = 1:Nc2

   % ���ݵر���ȥ��50%������������
   S = 0;
   for j = 1:round(pop/2)
      S = S + X(j,:).*fitness(j);      
   end

   Xcenter = S./((pop/2)*sum(fitness(1:round(pop/2))));
   %�������ĸ�����Ⱥ
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
    %�������
   [fitness, index]= sort(fitness);%����
   for j = 1:pop
      X(j,:) = X(index(j),:);
   end
   
    if(fitness(1) < GBestF)
       GBestF = fitness(1);
        GBestX = X(1,:);   
    end
   curve(Nc1 + t) = GBestF;
end
Best_pos =GBestX;
Best_score = curve(end);
end