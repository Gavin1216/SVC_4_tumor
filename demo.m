close all;
clear all;
clc;

addpath([cd '/utilies']);
load('Brain_Tumor1.mat') ;   
data=data';
RandIndex = randperm(size(data,2));
data=data(:,RandIndex);
Train_DAT=[];  
Test_DAT=[];
trls=[];
ttls=[];
nclass=5;
for i=0:4
    TEMP = data (2:5921,data(1,:)==i);
    Train_DAT=[Train_DAT TEMP(:,1:int8(size(TEMP,2)*0.8))];
    Test_DAT=[Test_DAT TEMP(:,int8(size(TEMP,2)*0.8)+1:size(TEMP,2))];
    trls=[trls repmat(i+1,[1,int8(size(TEMP,2)*0.8)])];
    ttls=[ttls repmat(i+1,[1,size(TEMP,2)-int8(size(TEMP,2)*0.8)])];
end
clear TEMP;
 



ID=[];
for time=1:50    
RandIndex1 = randperm(size(Train_DAT,2));
Train_DAT2=Train_DAT(:,RandIndex1);
trls2=trls(:,RandIndex1);
RandIndex2 = randperm(size(Train_DAT,1));
len=int32(size(Train_DAT2,1)/2);
Train_DAT2=Train_DAT2(RandIndex2(1:len),:);
Test_DAT2=Test_DAT(RandIndex2(1:len),:);
Train_DAT1=[]; 
trls1=[];
nclass=5;
for i=1:nclass
    TEMP = Train_DAT2(:,trls2(1,:)==i);
    Train_DAT1=[Train_DAT1 TEMP(:,1:int32(size(TEMP,2)*1))];
    trls1=[trls1 repmat(i,[1,int32(size(TEMP,2)*1)])]; 

end
clear TEMP;

opts.nClass        =   5;  %一共的类别数
opts.wayInit       =   'PCA'; %字典初始化方式
opts.lambda1       =   0.005; %
opts.lambda2       =   0;
opts.nIter         =  10;     %迭代次数
opts.show          =   true;
[Dict1,Drls1,CoefM1,CMlabel1] = FDDL3(Train_DAT1,trls1,opts);

lambda   =   0.005;
nClass   =   opts.nClass;
weight   =   0.5; 
td1_ipts.D    =   Dict1;
td1_ipts.tau1 =   lambda;
if size(td1_ipts.D,1)>=size(td1_ipts.D,2)
   td1_par.eigenv = eigs(td1_ipts.D'*td1_ipts.D,1);
else
   td1_par.eigenv = eigs(td1_ipts.D*td1_ipts.D',1);  
end
ID1   =   [];
for indTest = 1:size(Test_DAT2,2)
    fprintf(['Totalnum:' num2str(size(Test_DAT2,2)) 'Nowprocess:' num2str(indTest) '\n']);
    td1_ipts.y          =      Test_DAT2(:,indTest);   %取出当前列的测试样本
    [opts]              =      IPM_SC(td1_ipts,td1_par);
    s                   =      opts.x;%s代表系数
    for indClass  =  1:nClass
        temp_s            =  zeros(size(s));
        temp_s(indClass==Drls1) = s(indClass==Drls1);
        zz                =  Test_DAT2(:,indTest)-td1_ipts.D*temp_s;
        gap(indClass)     =  zz(:)'*zz(:);
        
        mean_coef_c         =   CoefM1(:,indClass);
        gCoef3(indClass)    =  norm(s-mean_coef_c,2)^2;    
    end
    
    wgap3  = gap + weight*gCoef3;
    index3 = find(wgap3==min(wgap3));
    id3    = index3(1);
    ID1     = [ID1 id3];
end  
ID=[ID;ID1];
end 

ID3=[];
for m=1:18
    ID3=[ID3 mode(ID(:,m))];
end
%}





fid = fopen(['result.txt'],'a');
fprintf(fid,'%s%8f\t\n','reco_rate2 = ',sum(ID3==ttls)/(length(ttls)));


