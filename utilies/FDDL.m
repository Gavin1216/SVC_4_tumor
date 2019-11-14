function [Dict,Drls,CoefM,CMlabel,P,cof,myIndex] = FDDL(TrainDat,TrainLabel,opts)
% ========================================================================
% Fisher Discriminative Dictionary Learning (FDDL), Version 1.0
% Copyright(c) 2011  Meng YANG, Lei Zhang, Xiangchu Feng and David Zhang
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
%
% This is an implementation of the algorithm for learning the
% Fisher Discriminative Dictionary from a labeled training data
%
% Please refer to the following paper
%
% Meng Yang, Lei Zhang, Xiangchu Feng, and David Zhang,"Fisher Discrimination 
% Dictionary Learning for Sparse Representation", In IEEE Int. Conf. on
% Computer Vision, 2011.
% 
%----------------------------------------------------------------------
%
%Input : (1) TrainDat: the training data matrix. 
%                      Each column is a training sample
%        (2) TrainDabel: the training data labels
%        (3) opts      : the struture of parameters
%               .nClass   the number of classes
%               .wayInit  the way to initialize the dictionary
%               .lambda1  the parameter of l1-norm energy of coefficient
%               .lambda2  the parameter of l2-norm of Fisher Discriminative
%               coefficient term
%               .nIter    the number of FDDL's iteration
%               .show     sign value of showing the gap sequence
%
%Output: (1) Dict:  the learnt dictionary via FDDL
%        (2) Drls:  the labels of learnt dictionary's columns
%        (2) CoefM: Mean Coefficient Matrix. Each column is a mean coef
%                   vector
%        (3) CMlabel: the labels of CoefM's columns.
%
%-----------------------------------------------------------------------
%
%Usage:
%Given a training data, including TrainDat and TrainLabel, and the
%parameters, opts.
%
%[Dict,CoefM,CMlabel] = FDDL(TrainDat,TrainLabel,opts)
%-----------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%
% normalize energy
%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%
%初始化降维矩阵
%%%%%%%%%%%%%%%%%%%%
%{
myIndex=[];   %即真正有用的维度。
trls=TrainLabel;
for i=1:opts.nClass 
    classi=TrainDat(:,trls==i);  %把该类取出来
    nclassi=TrainDat(:,trls~=i); %
    meani=mean(classi,2);
    nmeani=mean(nclassi,2);
    compare=abs(meani-nmeani);
    compare=compare./abs(nmeani);
    [B,index]=sort(compare,'descend');
    %numB=find(B>1);
    indexs=index(1:int32(size(classi,1)/opts.nClass),:);
    myIndex=[myIndex;indexs]; 
    
end 
myIndex=unique(myIndex,'rows');
%TrainDat=TrainDat(myIndex,:);
%TrainDat = TrainDat*diag(1./sqrt(sum(TrainDat.*TrainDat))); %归一化数据

nDim     = int32(size(TrainDat,1)/2);    % feature dimension
Tr_DAT=TrainDat;
Mean_Image = mean(Tr_DAT,2); %对训练数据每一行求平均，即所有样本的均值。
ZA = Tr_DAT-Mean_Image*ones(1,size(Tr_DAT,2));%即文中At.
ZB = []; %即文中的Ab
nClass=opts.nClass ;
for class=1:nClass
    Class_im   =   Tr_DAT(:,(trls==class)); % learn by each class     
    ZB(:,(trls==class))=repmat(mean(Class_im,2)-Mean_Image,1,size(Class_im,2));
end
%}

Avg=[];
trls=TrainLabel;
nclass=opts.nClass;
for i=1:nclass
    temp=TrainDat(:,trls==i);
    avg=[];
    col=size(temp,2);
    
    if(col>3)
        for j=1:size(temp,1)
        [Y1,~]=max(temp(j,:));
        [Y2,~]=min(temp(j,:));
        avg1=(sum(temp(j,:))-Y1-Y2)/(size(temp,2)-2);
        avg=[avg; avg1];
        end 
        Avg=[Avg avg];
      
    else 
    
        avg=mean(temp,2);
        Avg=[Avg avg];
    end
end
myIndex=[];
Avg=Avg;
for i=1:nclass 
    temp=TrainDat(:,trls==i); 
    meani=Avg(:,i);
    sum1=zeros(size(temp,1),1);
    for j=1:nclass
        if(j~=i)
            sum1=sum1+Avg(:,j);
        end
    end
    nmeani=sum1/(nclass-1);
    compare=abs(meani-nmeani); 
    compare=compare./abs(nmeani);
    [B,index]=sort(compare,'descend');
    numB=find(B>0.5);
    indexs=index(1:size(numB,1),:); 
    %indexs=index(1:int32(size(TrainDat,1)/nclass),:); 
    myIndex=[myIndex;indexs]; 
end
myIndex=unique(myIndex,'rows');
%TrainDat=TrainDat(myIndex,:);
nDim     = int32(size(TrainDat,1)/8);
%nDim     = 1000;
TrainDat = TrainDat*diag(1./sqrt(sum(TrainDat.*TrainDat))); %归一化数据
Avg=[];
for i=1:nclass
    Class_im   =   TrainDat(:,(trls==i));
    avg=mean(Class_im,2);
    Avg=[Avg avg];
end

ZA=[];
for i=1:nclass
    Class_im   =   TrainDat(:,(trls==i));
    meank=mean(Class_im,2);  %代表该类样本的平均值
    sum1=zeros(size(Class_im,1),1);
    for j=1:nclass
        if(j~=i)
            sum1=sum1+Avg(:,j);
        end
    end
    meanbk=sum1/(nclass-1); %代表其他类样本的均值
    CB(:,(trls==i))=repmat(meank-meanbk,1,size(Class_im,2));
    ZA1=Class_im-meanbk*ones(1,size(Class_im,2));
    ZA=[ZA ZA1];
end




[p,~]=Find_K_Max_Gen_Eigen(ZA*ZA'+CB*CB',eye(size(TrainDat,1)),nDim);%p为映射矩阵
%clear Tr_DAT;
%clear trls;

ptrainDat=p'*TrainDat;

size(CB)









%%%%%%%%%%%%%%%%%%
%initialize dict
%%%%%%%%%%%%%%%%%%
Dict_ini  =  []; 
Dlabel_ini = [];
for ci = 1:opts.nClass
    cdat          =    ptrainDat(:,TrainLabel==ci);
    dict          =    FDDL_INID(cdat,size(cdat,2),opts.wayInit);
    Dict_ini      =    [Dict_ini dict];
    Dlabel_ini    =    [Dlabel_ini repmat(ci,[1 size(dict,2)])];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialize coef without between-class scatter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ini_par.tau         =     opts.lambda1;
ini_par.lambda      =     opts.lambda2;
ini_ipts.D          =     Dict_ini;
coef = zeros(size(Dict_ini,2),size(TrainDat,2));
if size(Dict_ini,1)>size(Dict_ini,2)
      ini_par.c        =    1.05*eigs(Dict_ini'*Dict_ini,1);
else
      ini_par.c        =    1.05*eigs(Dict_ini*Dict_ini',1);
end
for ci =  1:opts.nClass
    fprintf(['Initializing Coef:  Class ' num2str(ci) '\n']);
    ini_ipts.X      =    ptrainDat(:,TrainLabel==ci);
    [ini_opts]      =    FDDL_INIC (ini_ipts,ini_par);
    coef(:,TrainLabel ==ci) =    ini_opts.A;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Main loop of Fisher Discriminative Dictionary Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 Fish_par.dls        =     Dlabel_ini;
 Fish_ipts.D         =     Dict_ini;
 Fish_ipts.trls      =     TrainLabel;
 Fish_par.tau        =     opts.lambda1;
 Fish_par.lambda2    =     opts.lambda2;
 
 Fish_nit            =     1;
 drls                =     Dlabel_ini;
 while Fish_nit<=opts.nIter  
    if size(Fish_ipts.D,1)>size(Fish_ipts.D,2)
      Fish_par.c        =    1.05*eigs(Fish_ipts.D'*Fish_ipts.D,1);
    else
      Fish_par.c        =    1.05*eigs(Fish_ipts.D*Fish_ipts.D',1);
    end
    %-------------------------
    %updating the coefficient
    %-------------------------
    for ci = 1:opts.nClass
        fprintf(['Updating coefficients, class: ' num2str(ci) '\n'])
        Fish_ipts.X         =  ptrainDat(:,TrainLabel==ci);
        Fish_ipts.A         =  coef;
        Fish_par.index      =  ci; 
        [Copts]             =  FDDL_SpaCoef (Fish_ipts,Fish_par);
        coef(:,TrainLabel==ci)    =  Copts.A;
        CMlabel(ci)         =  ci;
        CoefM(:,ci)         =  mean(Copts.A,2);
    end
    [GAP_coding(Fish_nit)]  =  FDDL_FDL_Energy(ptrainDat,coef,opts.nClass,Fish_par,Fish_ipts);
    
    %------------------------
    %updating the dictionary
    %------------------------
    for ci = 1:opts.nClass
        fprintf(['Updating dictionary, class: ' num2str(ci) '\n']);     
        [Fish_ipts.D(:,drls==ci),Delt(ci).delet]= FDDL_UpdateDi (ptrainDat,coef,...
            ci,opts.nClass,Fish_ipts,Fish_par);
    end
    [GAP_dict(Fish_nit)]  =  FDDL_FDL_Energy(ptrainDat,coef,opts.nClass,Fish_par,Fish_ipts);
    
    
    
    
    
    newD = []; newdrls = []; newcoef = [];
    for ci = 1:opts.nClass
        delet = Delt(ci).delet;
        if isempty(delet)
           classD = Fish_ipts.D(:,drls==ci);
           newD = [newD classD];
           newdrls = [newdrls repmat(ci,[1 size(classD,2)])];
           newcoef = [newcoef; coef(drls==ci,:)];
        else
            temp = Fish_ipts.D(:,drls==ci);
            temp_coef = coef(drls==ci,:);
            for temp_i = 1:size(temp,2)
                if sum(delet==temp_i)==0
                    newD = [newD temp(:,temp_i)];
                    newdrls = [newdrls ci];
                    newcoef = [newcoef;temp_coef(temp_i,:)];
                end
            end
        end
    end
    
    
     %%%%%%%%%%%%%%%%%%
    %此处更新降维矩阵P
    %%%%%%%%%%%%%%%%%%
    %求解问题  Min{ ||P*train_dat(i) - D*Xi ||f2 +}
    
   % if Fish_nit<opts.nIter
        
       beta = 0.008;%     iter_num_sub= 1;
       gamma1=10;
       gamma2=1;

       for classi=1:opts.nClass 
           Ai=TrainDat(:,TrainLabel==classi);
           Xi=newcoef(:,TrainLabel==classi);
           Di=newD(:,newdrls==classi);
           Xii=newcoef(TrainLabel==classi,TrainLabel==classi);
           phi=((Ai-p*(newD*Xi))*(Ai-p*(newD*Xi))') + ((Ai-p*(Di*Xii))*(Ai-p*(Di*Xii))');
           Class_im   =   TrainDat(:,(TrainLabel==i));
           sum1=zeros(size(Class_im,1),1);
            for j=1:nclass
                if(j~=i)
                    sum1=sum1+Avg(:,j);
                end
            end
        meanbk=sum1/(nclass-1); %代表其他类样本的均值
        ZA=Class_im-meanbk*ones(1,size(Class_im,2));
        CB=(Avg(:,classi)-meanbk)*ones(1,size(Class_im,2));
     numcomps = size(p,2);
     S=gamma1*(ZA*ZA')+gamma2*(CB*CB');
    
    
          [U,D] = eig(single(phi-S)); 
          D=diag(D);
          [~, I] = sort(D, 'descend');
           U=U(:, I(1 : numcomps));
          p1 = U;
          p = p+beta*(p1-p);
          fprintf(['Updating P, class: ' num2str(classi) '\n']);
       end
       
       %{
       for k=1:3
          phi=zeros(size(TrainDat,1),size(TrainDat,1));    
       for classi=1:opts.nClass
           Ai=TrainDat(:,TrainLabel==classi);
           Xi=newcoef(:,TrainLabel==classi);
           Di=newD(:,newdrls==classi);
           Xii=newcoef(TrainLabel==classi,TrainLabel==classi);
           phi=phi+((Ai-p*(newD*Xi))*(Ai-p*(newD*Xi))') + ((Ai-p*(Di*Xii))*(Ai-p*(Di*Xii))');
       end
     numcomps = size(p,2);
     S=gamma1*(ZA*ZA')+gamma2*(CB*CB');

    
          [U,D] = eig(single(phi-S)); 
          D=diag(D);
          [~, I] = sort(D, 'descend');
           U=U(:, I(1 : numcomps));
          p1 = U;
          p = p+beta*(p1-p);
          fprintf(['Updating P, time: ' num2str(k) '\n']); 
       end
       %}
   % end
    
    
    
     
    
    
    
    
    Fish_ipts.D  = newD;
    coef         = newcoef;
    drls         = newdrls;
    ptrainDat=p'*TrainDat;
    Fish_par.dls        =     drls;
    
    Fish_nit = Fish_nit +1;
    end
    
    Dict = Fish_ipts.D;
    Drls = drls;
    cof=coef;
    P=p;
    if opts.show
    subplot(1,2,1);plot(GAP_coding,'-*');title('GAP_coding');
    subplot(1,2,2);plot(GAP_dict,'-o');title('GAP_dict'); 
    end
return;