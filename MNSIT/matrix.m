function f = matrix(X) 
a = load('A.mat');
A=a.A;
b = load('B.mat');
B=b.B;
f = dot(A,X)+0.5*X*B*(X');


  
