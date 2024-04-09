% 注意本文件处理的数据对象必须均为.mat文件
clear
data_list(1).name='data1';
data_list(1).varname='Data1_AI_*';
data_list(2).name='data2';
data_list(2).varname='data';
data_list(3).name='data3/Data';
data_list(3).varname='*';
data_list(4).name='data4';
data_list(4).varname='*';
snr_list=[20 40];
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
                seq=eval(cell2mat(var_list(c)));
                
                m=find(abs(seq-mean(seq))>5*std(seq));
                if all(data_list(a).name=='data1') && contains(file_list(b).name,'current') && ~isempty(m)
                    seq1=seq(1:m(1));
                    seq1_minmax=2*(seq1-min(seq1))/(max(seq1)-min(seq1))-1;
                    seq2=seq(m(1):end);
                    seq2_minmax=2*(seq2-min(seq2))/(max(seq2)-min(seq2))-1;
                    noise1=awgn(seq1_minmax,d);
                    noise1=(noise1+1)/2*(max(seq1)-min(seq1))+min(seq1);
                    noise2=awgn(seq2_minmax,d);
                    noise2=(noise2+1)/2*(max(seq2)-min(seq2))+min(seq2);
                    noise=[noise1;noise2];
                else
                    seq_minmax=2*(seq-min(seq))/(max(seq)-min(seq))-1;
                    noise=awgn(seq_minmax,d);
                    noise=(noise+1)/2*(max(seq)-min(seq))+min(seq);
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

