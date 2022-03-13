function [sigma,Prediction,weights] = Q2RBFN_train(sigma,x1,x2,y,lambda, ...
    sigma_flag)
x1 = double(x1);
x2 = double(x2);
y = double(y');
N = size(x1,2);
M = size(x2,2);
dis = zeros(N,M);
dis_2 = zeros(M,M);
if sigma_flag == 1
    for i = 1:M
        for  j = 1:M
            dis_2(i,j) = sqrt(sum((x2(:,i) - x2(:,j)).^2,'all'));
        end
    end
    sigma = max(dis_2,[],'all')/sqrt(2*M);
end

for i = 1:N
    for j = 1:M
        dis(i,j) = sum((x1(:,i) - x2(:,j)).^2);
    end
end
F = exp(- dis/(2* sigma^2));
input = [F y(:,1)];
weights = LMS(input,2000,0.001,lambda);
Prediction = [ones(N,1) F] * weights;