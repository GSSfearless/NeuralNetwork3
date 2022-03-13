function [y, erorr] = RBFN_test(sigma, x1, x2,weights,y2)


x1 = double(x1');
x2 = double(x2');
y2 = double(y2');
L1 = length(x1);      % L1 = 41
L2 = length(x2);      % L2 = 321
dis = zeros(L1,L2);    % dis = 41x321 L1xL2
for i = 1:L1
    for j = 1:L2
        dis(i,j) = sum((x1(i,:) -x2(j,:)).^2);
    end
end
F = exp(- dis/(2* sigma^2));   % F = 41x321  L1xL2


y = [ones(L2,1) F'] * weights;   % weights 42x1  


erorr = sum((y - y2).^2);


