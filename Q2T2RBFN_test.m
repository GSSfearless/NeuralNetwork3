function [y, erorr] = Q2T2RBFN_test(sigma, x1, x2,weights,y2)
L1 = size(x1,2);      % L1 column of x1
L2 = size(x2,2);      % L2 column of x2
x1 = double(x1');
x2 = double(x2');
y2 = double(y2);
dis = zeros(L1,L2);    % dis = L1xL2
for i = 1:L1
    for j = 1:L2
        dis(i,j) = sum((x1(i,:) -x2(j,:)).^2);
    end
end
F = exp(- dis/(2* sigma^2));   % F =  L1xL2

if L2 ~= length(weights)
   y = [ones(L1,1) F] * weights;   % F' = L2xL1 weights 42x1  
else
    y = F * weights;
end

erorr = sum((y - y2').^2);