function [weights] = LMS(data,iter,lr,lambda)

s = size(data);
m = s(1); 
n = s(2) -1; 
weights = rand(n+1,1)*2-1;
X = [ones(m,1) data(:,1:n)];
D = data(:,s(2));
loss = zeros(iter,2);
early_stop = 0;
mse_min = 1e4;

for i = 1:iter
    e = D - X*weights;
    grad = -(X' * e + lambda * weights);
    weights = weights - lr * grad;
    mse(e)
    if mse(e)<= 1e-6
        break
    end
    if mse(e) < mse_min
        mse_min = mse(e);
    elseif mse(e) >= mse_min
        early_stop = early_stop + 1;
    end
    if early_stop >= 1
        break
    end
end
     
    