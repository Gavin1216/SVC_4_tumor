%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Eigen_Vector,Eigen_Value]=Find_K_Max_Gen_Eigen(Matrix1,Matrix2,Eigen_NUM)

[NN,NN]=size(Matrix1); %NN代表矩阵列数
[V,S]=eig(Matrix1,Matrix2); %Note this is equivalent to; [V,S]=eig(St,SL); also equivalent to [V,S]=eig(Sn,St); %V是特征向量，
%[V,D]=eig(A,B)：由eig(A,B)返回方阵A和B的N个广义特征值，构成N×N阶对角阵D，其对角线上的N个元素即为相应的广义特征值，同时将返回相应的特征向量构成N×N阶满秩矩阵，且满足AV=BVD。%

S=diag(S);
[S,index]=sort(S);

Eigen_Vector=zeros(NN,Eigen_NUM);
Eigen_Value=zeros(1,Eigen_NUM);

p=NN;
for t=1:Eigen_NUM
    Eigen_Vector(:,t)=V(:,index(p));
    Eigen_Value(t)=S(p);
    p=p-1;
end