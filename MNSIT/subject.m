function [c,ceq] = subject(X)
p=importdata('L2.txt');
c =norm(X,2)-p;
ceq=[];


