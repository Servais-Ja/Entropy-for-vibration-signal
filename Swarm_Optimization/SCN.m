%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stochastic Configuration Netsworks Class (Matlab)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2017
classdef SCN
    properties%类的属性，没有赋值默认为空即'[]'
        Name = 'Stochastic Configuration Networks';
        version = '1.0 beta';
        % Basic parameters (networks structure)
        L       % hidden node number / start with 1
        W       % input weight matrix
        b       % hidden layer bias vector
        Beta    % output weight vector
        % Configurational parameters
        r       % regularization parameter 正则化参数
        tol     % tolerance
        Lambdas % random weights range, linear grid search 随机权值范围（线性网格搜索）
        L_max   % maximum number of hidden neurons 隐藏节点的最大个数
        T_max   % Maximum times of random configurations 随机配置的最大次数（每次随机生成的节点的数目）
        % Else
        nB = 1  % how many node need to be added in the network in one loop 一个循环需要在网络中增加多少个节点
        verbose = 50 % display frequency
        COST = 0% final error
    end
    %% Funcitons and algorithm
    methods%类的方法
        %% Initialize one SCN model
        function obj = SCN(L_max, T_max, tol, Lambdas, r) % 在构造函数中进行属性变量初始化（默认值）
            format long; %改为显示有效数字16位
            
            obj.L = 1;
  
            if ~exist('L_max', 'var') || isempty(L_max)
                obj.L_max = 100;
            else
                obj.L_max = L_max;
                if L_max > 5000
                    obj.verbose = 500; % does not need too many output
                end
            end
            if ~exist('T_max', 'var') || isempty(T_max)
                obj.T_max=  100;
            else
                obj.T_max = T_max;
            end
            if ~exist('tol', 'var') || isempty(tol)
                obj.tol=  1e-4;
            else
                obj.tol = tol;
            end
            if ~exist('Lambdas', 'var') || isempty(Lambdas)
                obj.Lambdas=  [0.5, 1, 3, 5, 7, 9, 15, 25, 50, 100, 150, 200];
            else
                obj.Lambdas = Lambdas;
            end
            if ~exist('r', 'var') || isempty(r)
                obj.r =  [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.99999];
            else
                obj.r = r;
            end
        end
                
        %% inequality equation return the ksi
        function  [obj, ksi] = InequalityEq(obj, eq, gk, r_L) % eq是所有特征组成的矩阵的第m列，gk是t隐藏节点的所有输出，r_L是一个正则化参数
            ksi = ((eq'*gk)^2)/(gk'*gk) - (1 - r_L)*(eq'*eq);
        end
        %% 计算节点优劣的函数
        function Ksi_t = F_node_score(obj, WTbT)%这里的x应该是WT, bT, i_r
            global fun_x
            global fun_e
            global d
            global m
            WTbT=WTbT';
            WT=WTbT(1:d,1);
            bT=WTbT(d+1:d+m,1);
            X = fun_x;
            E0 = fun_e;
            HT = logsig(bsxfun(@plus, X*WT, bT));
            for i_r = 1:length(obj.r)%???怎么找最佳的正则化参数要不要找
                r_L = obj.r(i_r);
                for i_m = 1:m                            
                    eq = E0(:,i_m);         % 所有特征组成的矩阵的第m列
                    gk = HT;
                    [obj, ksi_m(i_m)] = obj.InequalityEq(eq, gk, r_L);% 这里的Inequality就是评价节点对预测准确结果贡献程度的函数
                end
                if min(ksi_m) > 0                 % 如果满足min(ksi_m) > 0的隐藏节点数>= obj.nB就不用进行接下来的正则化的迭代，也就同时停止权值的迭代
                    Ksi_t = sum(ksi_m);         % 这里的Ksi_t表示这个节点的优劣
                    break;                      % r loop
                else
                    Ksi_t = -Inf;
                    continue;
                end
            end
        end
        %% Search for {WB,bB} of nB nodes using BSO
        function [WB, bB, Flag] = BSO_Search(obj, X, E0)
            Flag =  0;                              % 0: continue; 1: stop; return a good node /or stop training by set Flag = 1
            global d
            global m
            [~,d] = size(X);                        % Get Sample and feature number 特征和标签分别用多少位表示，相当于每一行是一个个体；其中~表示忽略输出参数
            [~,m] = size(E0);
            % Linear search for better nodes
            C = []; % container of kesi
            global fun_x
            global fun_e
            fun_x=X;
            fun_e=E0;
            %%%BSO
            [WBbB,Ksi_t,~]=BSO(obj.F_node_score,d,beetle_number,Max_iter,x_boundary,v_boundary);
            WB=WBbB(1:d,1);
            bB=WBbB(d+1:d+m,1);
            if Ksi_t == -Inf                 % discard w b 如果没有没有找到节点或要求数量的节点（提前停止迭代），丢弃w b
                disp('End Searching ...');
                Flag = 1;
            end
        end
        %% Search for {WB,bB} of nB nodes
        function [WB, bB, Flag] = SC_Search(obj, X, E0)
            Flag =  0;                              % 0: continue; 1: stop; return a good node /or stop training by set Flag = 1
            WB  = [];
            bB  = [];
            [~,d] = size(X);                        % Get Sample and feature number 特征和标签分别用多少位表示，相当于每一行是一个个体；其中~表示忽略输出参数
            [~,m] = size(E0);
            % Linear search for better nodes
            C = []; % container of kesi
            for i_Lambdas = 1: length(obj.Lambdas)  % i index lambda 对所有权值；其中Lambda是所有隐藏节点的输出权重
                                                    % ???这样按lambda成比例改变搜索范围和文章里的分层寻找的方法不太一样
                                                    % ???分层随机生成也应该全部进行完呀，不应该该跳过吧
                Lambda = obj.Lambdas(i_Lambdas);    % Get the random weight and bias range 得到随机权重和偏置范围
                
                % Generate candidates T_max vectors of w and b for selection
                % WT和bT应该就是我要调整的参数，这是参数的初始化
                WT = Lambda*( 2*rand(d, [obj.T_max])-1 ); % WW is d-by-T_max 每一列是一个节点对应的参数；Lamdas的改变其实是取得的WT的范围
                bT = Lambda*( 2*rand(1, [obj.T_max])-1 ); % bb is 1-by-T_max
                
                HT = logsig(bsxfun(@plus, X*WT, bT)); % 神经网络的前向传播过程；bsxfun对矩阵A和B进行指定计算；并自动扩维
                %HT每一行就是一个对象通过隐藏层后，每个隐藏节点的输出，激活函数是sigmoid
                for i_r = 1:length(obj.r)           % 对于每一个正则化参数
                    r_L = obj.r(i_r);               % get the regularization value
                    % calculate the Ksi value
                    for t = 1: obj.T_max            % searching one by one 对每一个隐藏节点
                        % Calculate H_t
                        H_t = HT(:,t);              % t隐藏节点的所有输出
                        % Calculate kesi_1 ... kesi_m
                        % 评价节点优劣
                        ksi_m = zeros(1, m);
                        for i_m = 1:m                            
                            eq = E0(:,i_m);         % 所有特征组成的矩阵的第m列
                                                    % 这里的E0就是残差向量
                            gk = H_t;
                            [obj, ksi_m(i_m)] = obj.InequalityEq(eq, gk, r_L);% 这里的Inequality就是评价节点对预测准确结果贡献程度的函数???这里的meu不算了吗
                        end
                        Ksi_t = sum(ksi_m);         % 这里的Ksi_t表示这个节点的优劣
                                                    % 根据Ksi_t选择最佳节点的过程在后面
                        if min(ksi_m) > 0
                            C = cat(2, C, Ksi_t);   % 每一位标签对应的ksi之和和相应的节点t的参数的水平堆叠
                            WB  = cat(2, WB, WT(:,t));
                            bB  = cat(2, bB, bT(:,t));
                        end
                    end
                    nC = length(C);
                    if nC >= obj.nB                 % 如果满足min(ksi_m) > 0的隐藏节点数>= obj.nB就不用进行接下来的正则化的迭代，也就同时停止权值的迭代
                        break;                      % r loop
                    else
                        continue;
                    end
                end %(r)
                if nC >= obj.nB
                    break; % lambda loop
                else
                    continue;
                end
            end % (lambda)
            % Return the good node / or stop the training.
            if nC>= obj.nB                          % 如果是（提前停止迭代了）找到节点了
                [~, I] = sort(C, 'descend');        % 降序排序，I为索引值
                I_nb = I(1:obj.nB);                 % 选择最佳的nB个节点
                WB = WB(:, I_nb);                   % 得到最佳节点对应的参数
                bB = bB(:, I_nb);
                %HB = HB(:, I_nb);
            end
            if nC == 0 || nC<obj.nB                 % discard w b 如果没有提前停止迭代，丢弃w b
                disp('End Searching ...');
                Flag = 1;
            end
        end
        %% Add nodes to the model
        function obj = AddNodes(obj, w_L, b_L)
            obj.W = cat(2,obj.W, w_L);
            obj.b = cat(2,obj.b, b_L);
            obj.L = length(obj.b);
        end
        
        %% Compute the Beta, Output, ErrorVector and Cost
        function [obj, O, E, Error] = UpgradeSCN(obj, X, T)%该函数中O即是对每个样本的输出，E是所有模型输出和真实值的差值
            H = obj.GetH(X);                        % 得到隐藏层输出
            obj = obj.ComputeBeta(H,T);             % ???前后obj会不会不一样了？
            O = H*obj.Beta;
            E = T - O;
            Error =  Tools.RMSE(E);
            obj.COST = Error;
        end     
        
        %% ComputeBeta
        function [obj, Beta] = ComputeBeta(obj, H, T)
            Beta = pinv(H)*T;                       % 求伪逆矩阵
            obj.Beta = Beta;                        % 略显多余
        end              
        %% Regression
        function [obj, per] = Regression(obj, X, T)             
            per.Error = [];
            E = T;
            Error =  Tools.RMSE(E);
            % disp(obj.Name);
            while (obj.L < obj.L_max) && (Error > obj.tol)            
                if mod(obj.L, obj.verbose) == 0
                    fprintf('L:%d\t\tRMSE:%.6f \r', obj.L, Error );
                end
                [w_L, b_L, Flag] = SC_Search(obj, X, E);% Search for candidate node / Hidden Parameters
                if Flag == 1
                    break;% could not find enough node
                end
                obj = AddNodes(obj, w_L, b_L);                 
                [obj, ~ , E, Error ] = obj.UpgradeSCN(X, T); % Calculate Beta/ Update all                
                %log
                per.Error = cat(2, per.Error, repmat(Error, 1, obj.nB));
            end% while
            fprintf('#L:%d\t\tRMSE:%.6f \r', obj.L, Error );
            disp(repmat('*', 1,30));
        end
        
        %% Classification
        function [obj, per] = Classification(obj, X, T, Method)             % X为输入量即特征，T为输出量即标签
            per.Error = []; % Cost function error
            per.Rate = [];  % Accuracy Rate
            E = T;
            Error =  Tools.RMSE(E);                                 % ???均方根误差
            Rate = 0;
            % disp(obj.Name);                                         % 显示Name属性
            while (obj.L < obj.L_max) && (Error > obj.tol)          % 循环终止条件：误差小于容忍度，隐层节点个数超过设定的最大值，SC_Search找不到足够的节点了
                                                                    % L是隐藏层节点数，从1开始
                if mod(obj.L, obj.verbose) == 0                     % 节点数是否被verbose整除，若整除：显示目前的节点数，误差，准确率
                    % fprintf('L:%d\t\t RMSE:%.6f; \t\tRate:%.2f\r', obj.L, Error, Rate);
                end
                if Method==0                                        % 如果Method为0则用传统的SCN方法生成节点
                    [w_L, b_L, Flag] = SC_Search(obj, X, E);        % 找最佳节点参数，每次看T_max个节点
                else
                    [w_L, b_L, Flag] = BSO_Search(obj, X, E);
                end
                if Flag == 1
                    break;                                          % could not find enough node
                end
                obj = AddNodes(obj, w_L, b_L);                      % 把这个找到的最佳节点加进去
                [obj, O, E, Error ] = obj.UpgradeSCN(X, T);         % Calculate Beta/ Update all 根据目前的SCN更新E值、误差，其中T是真实标签
                Rate = 1- confusion(T',O');
                % Training LOG
                per.Error = cat(2, per.Error, repmat(Error, 1, obj.nB));
                per.Rate = cat(2, per.Rate,  repmat(Rate, 1, obj.nB));
            end% while
            % fprintf('#L:%d\t\t RMSE:%.6f; \t\tRate:%.2f\r', obj.L, Error, Rate);
            % disp(repmat('*', 1,30));
        end

        %% Output Matrix of hidden layer
        function H = GetH(obj, X) % 略显多余
            H =  obj.ActivationFun(X);
        end
        %% Sigmoid function
        function H = ActivationFun(obj,  X) % 得到的H的每一行是对该行对应的样本个体目前加入节点的每个节点的输出
            disp(size(X))
            disp(size(obj.W))
            H = logsig(bsxfun(@plus, X*[obj.W],[obj.b]));              
        end
        %% Get Output
        function O = GetOutput(obj, X)
            H = obj.GetH(X);
            O = H*[obj.Beta];
        end
        %% Get Label
        function O = GetLabel(obj, X)
            O = GetOutput(obj, X);
            O = Tools.OneHotMatrix(O);
        end
        %% Get Accuracy
        function [Rate, O] = GetAccuracy(obj, X, T)
            O = obj.GetLabel(X);
            Rate = 1- confusion(T',O');
        end
        %% Get Error, Output and Hidden Matrix
        function [Error, O, H, E] = GetResult(obj, X, T)
            % X, T are test data or validation data
            H = obj.GetH(X);
            O = H*(obj.Beta);
            E = T - O;
            Error =  Tools.RMSE(E);
        end
 
    end % methods
end % class
