function temp = PointToPlaneDistance(x,x1,x2,x3);
%Input: the query example x, training examples x1, x2 and x3. Each sample is a row vector 
%Output: the distance between x and the plane passing through x1, x2 and x3 
%Reference: the report of Shangzhi Li
%Reference: J. T. Chien and C. C. Wu, "Discriminant waveletfaces and nearest feature classifiers for face recognition," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 24, pp. 1644-1649, Dec 2002.
%This code is written by Gui Jie on the morning of 12/05/2012.
%If you have find some bugs in the codes, feel free to contract me

a1 = x2-x1;
a2 = x3-x1;
c = x-x1;
A = [a1; a2];
temp1 = inv(A*A')*(A)*(c');
p = (temp1')*A;
temp = norm(c-p,2);