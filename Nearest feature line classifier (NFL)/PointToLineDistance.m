function temp = PointToLineDistance(x,x1,x2,x3);
%Input: the query example x, training examples x1 and x2. Each sample is a row vector 
%Output: the distance between x and the straight line passing through x1
%and x2 
%Reference: S. Z. Li and J. W. Lu, "Face recognition using the nearest
%feature line method," IEEE Transactions on Neural Networks, vol. 10, pp. 439-443, Mar 1999.
%This code is written by Gui Jie on the evening 12/05/2012.
%If you have find some bugs in the codes, feel free to contract me

mu = (((x2-x1)*(x-x1)'))/((x2-x1)*((x2-x1)'));% The sixth line of left part of P441 of the reference
p = x1+mu*(x2-x1);% The first line of left part of P441 of the reference
temp = norm(x-p,2);%Eq. (1) of the reference