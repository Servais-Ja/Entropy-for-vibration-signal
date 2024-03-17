% 注意本文件处理的数据对象必须均为.mat文件
clear
data_list(1).name='data1';
data_list(1).varname='Data1_AI_*';
data_list(2).name='data2';
data_list(2).varname='data';
snr_list=[-20 -15 -10 -5 0 5];
for a=1:length(data_list)
    datafolder=data_list(a).name;
    file_list=dir([datafolder,'/*.mat']);
    file_len=length(file_list);
    for b=1:file_len
        load([datafolder,'/',file_list(b).name])
        var_list=who(data_list(a).varname);
        var_len=length(var_list);
        for d=snr_list
            for c=1:var_len
                noise=awgn(eval(cell2mat(var_list(c))),d);
                [xx,yy]=size(eval(cell2mat(var_list(c))));
                if xx>yy
                    noise=noise';
                end
                clearvars(cell2mat(var_list(c)));
                eval([cell2mat(var_list(c)) '= noise;']);
            end
            folder=[datafolder,'_',num2str(d)];
            if exist(folder)==0 %%判断文件夹是否存在
                mkdir(folder);  %%不存在时候，创建文件夹
            end
            save([folder '/' file_list(b).name])
        end
        clearvars -except data_list snr_list file_list file_len a datafolder
    end
end

