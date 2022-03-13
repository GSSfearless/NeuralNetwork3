function [center] = Kmeans(data,center)
data = double(data);
center = double(center);
L1 = size(data,2);
L2 = size(center,2);
dis = zeros(L1,L2);
label = zeros(L1,1);
label_flag = label;
for iter = 1 : 200
    for i = 1: L1
        for j = 1:L2
            dis(i,j) = sqrt(sum((data(:,i) - center(:,j)).^2,'all'));
        end
        if dis(i,1) < dis(i,2)
            label(i) = 0;
        else
            label(i) = 1;
        end
    end
    idx_0 = find( label ==0);
    idx_1 = find( label ==1);
%     idx_0 = find(label == 4);
%     idx_1 = find(label == 6);
    x0 = data(:,idx_0);
    x1 = data(:,idx_1);
    m0 = mean(x0,1);
    m1 = mean(x1,1);
    center = [m0  m1];
    if sum(abs(label - label_flag))==0
        break
    else
        label_flag  = label;
    end
end







