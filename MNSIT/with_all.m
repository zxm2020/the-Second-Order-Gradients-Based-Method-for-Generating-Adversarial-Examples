A=xlsread('(1).xlsx');
save A;
B=xlsread('(2).xlsx');
save B;
a = [];
b = [];
Aeq = [];
beq = [];
lb = ones(1,784)*(-1);
ub = ones(1,784);
x0 = ones(1,784)*(-1);
options = optimoptions(@fmincon,'Display','iter','MaxIterations',1000000,'MaxFunctionEvaluations',100000);
[x,fval,exitflag,output] = fmincon('matrix',x0,a,b,Aeq,beq,lb,ub,'subject',options);
save x