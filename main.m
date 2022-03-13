%%%%%%%%%%%%% Gan Runze %%%%%%%%%%%%%%%%%%%

%%
clear; clc; close all;

x1 = -1.6:0.08:1.6; %training data
x2 = -1.6:0.01:1.6; %testing data
y1 = 1.2*sin(pi*x1) - cos(2.4*pi*x1);
y2 = 1.2*sin(pi*x2) - cos(2.4*pi*x2);
L = length(y1);
y_noise = y1 + 0.3* randn([1,L]);
lambda = 0;
sigma_flag = 0;
sigma = 0.1;
[sigma, Prediction, weights] = RBFN_train(sigma,x1,x1,y_noise,lambda, ...
    sigma_flag);
%%
[output, error] = RBFN_test(sigma, x1, x2,weights,y2);


plot(x1,y1,'r','LineWidth',1.5);
hold on;
plot(x1,Prediction,'b','LineWidth',1.5);
xlabel('X value');
ylabel('Y value');
legend('Ground Truth','Approximate Value');
title('RBFN training accuracy');

figure;
plot(x2,y2,'r','LineWidth',1.5);
hold on;
plot(x2,output,'b','LineWidth',1.5);
hold on;
xlabel('X value');
ylabel('Y value');
legend('Ground Truth','Approximate Value');
title('RBFN testing accuracy');

%%  Q1 b
clear; clc; close all;
x1 = -1.6:0.08:1.6;
y1 = 1.2*sin(pi*x1) - cos(2.4*pi*x1);
x2 = -1.6:0.01:1.6;
y2 = 1.2*sin(pi*x2) - cos(2.4*pi*x2);
center_1 = x1(randperm(41,41));
center_2 = x2(randperm(321,321));
L = length(y1);
sigma = 0.1;
lambda = 0;
sigma_flag = 0;
[sigma, Prediction, weights] = RBFN_train(sigma,x1,center_1,y1, lambda, ...
    sigma_flag);
plot(x1,y1,'r','LineWidth',1.5);
hold on;
plot(x1,Prediction,'b','LineWidth',1.5);
xlabel('X value');
ylabel('Y value');
legend('Ground Truth','Approximate Value');
title('RBFN training accuracy with Fixed Centers Selected at Random');
%%

[output,error] = RBFN_test(sigma,x1,center_2,weights,y2);
figure;
plot(x2,y2,'r','LineWidth',1.5);
hold on;
plot(x2,output,'b','LineWidth',1.5);
xlabel('X value');
ylabel('Y value');
legend('Ground Truth','Approximate Value');
title('RBFN testing accuracy with Fixed Centers Selected at Random');


%%
clear; clc; close all;
x1 = -1.6:0.08:1.6;
y1 = 1.2*sin(pi*x1) - cos(2.4*pi*x1);
x2 = 1.6:0.01:1.6;
y2 = 1.2*sin(pi*x2) - cos(2.4*pi*x2);
L = length(y1);
y1_noise = y1 + 0.3*randn([1,L]);
sigma = 0.1;
lambda = [0, 0.1, 0.2, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0,100];
sigma_flag = 0;
result = [];
error = [];
for i = 1:length(lambda)
    [sigma, Prediction,weights] = RBFN_train(sigma,x1,x1,y1_noise,lambda(i),sigma_flag);
    [output,e] = RBFN_test(sigma,x1,x2,weights,y2);
    result = [result output];
    error = [error e];
    figure;
    plot(x1,y1,'r','LineWidth',1.5);
    hold on;
    plot(x1,Prediction,'b','LineWidth',1.5);
    xlabel('X value');
    ylabel('Y value');
    title('Training set with Lambda =',lambda(i));
    legend('Ground Truth','Approximate Value');

end
    

%% Q2
clear; clc; close all
load mnist_m.mat;
% Hand writing class: 4,6
% train set
trainIdx_1 = find( train_classlabel==4 | train_classlabel==6);
train_label_1 = train_classlabel(trainIdx_1);
train_label_1(train_label_1(:,:)==4 | train_label_1(:,:)==6)=1;
train_data_1 = train_data(:,trainIdx_1);

trainIdx_0 = find( train_classlabel==0 | train_classlabel==1 | train_classlabel==2 | train_classlabel==3 | train_classlabel==5 |train_classlabel==7 | train_classlabel==8 | train_classlabel==9);
train_label_0 = train_classlabel(trainIdx_0);
train_label_0(train_label_0(:,:)==0 | train_label_0(:,:)==1 | train_label_0(:,:)==2 | train_label_0(:,:)==3 | train_label_0(:,:)==5 | train_label_0(:,:)==7 | train_label_0(:,:)==8 | train_label_0(:,:)==9) = 0;
train_data_0 = train_data(:,trainIdx_0);

train_label = [train_label_0 train_label_1];
train_data = [train_data_0 train_data_1];
% test data
testIdx_1 = find( test_classlabel==4 | test_classlabel==6);
test_label_1 = test_classlabel(testIdx_1);
test_label_1(test_label_1(:,:)==4 | test_label_1(:,:)==6)=1;
test_data_1 = test_data(:,testIdx_1);

testIdx_0 = find( test_classlabel==0 | test_classlabel==1 | test_classlabel==2 | test_classlabel==3 | test_classlabel==5 | test_classlabel==7 | test_classlabel==8 | test_classlabel==9);
test_label_0 = test_classlabel(testIdx_0);
test_label_0(test_label_0(:,:)==0 | test_label_0(:,:)==1 | test_label_0(:,:)==2 | test_label_0(:,:)==3 | test_label_0(:,:)==5 | test_label_0(:,:)==7 | test_label_0(:,:)==8 | test_label_0(:,:)==9) = 0;
test_data_0 = test_data(:,testIdx_0);

test_label = [test_label_0 test_label_1];
test_data = [test_data_0 test_data_1];


lambda_list = [0, 0.1, 0.2, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100]; % 10
sigma_list = [0, 0.1, 1.0, 10.0, 100, 1000, 10000]; % 7
sigma_flag=0;

%%%% Here we select which question to solve %%%%%%%%%%%%%%%%
question = 1; 

% 1
if question == 1
    lambda = 0;
    sigma = 100;
    [sigma, TrPred,weights] = Q2RBFN_train(sigma,train_data,train_data,train_label,lambda,sigma_flag);
    sigma = 100;
    %[TePred,error] = Q2RBFN_test(sigma,train_data,test_data,weights,test_label);
    [TePred,error] = Q2T2RBFN_test(sigma,test_data,train_data,weights,test_label);
    TrAcc = zeros(1,1000);
    TeAcc = zeros(1,1000);
    thr = zeros(1,1000);
    TrLabel = train_label;
    TeLabel = test_label;
    TrN = length(TrLabel);
    TeN = length(TeLabel);
    for i = 1:1000
        t = (max(TrPred)-min(TrPred)) * (i-1)/1000 + min(TrPred);
        thr(i) = t;
        TrAcc(i) = (sum(TrLabel(TrPred<t)==0) + sum(TrLabel(TrPred>=t)==1)) / TrN;
        TeAcc(i) = (sum(TeLabel(TePred<t)==0) + sum(TeLabel(TePred>=t)==1)) / TeN;
    end
    plot(thr,TrAcc,'.- ',thr,TeAcc,'^-');legend('tr','te');
    title('lambda :',num2str(lambda));


elseif question == 2
    center = train_data(:,randperm(600,2));
    sigma = 10000;
    lambda = 0;
    sigma_flag = 1;
    [sigma,TrPred,weights] = Q2RBFN_train(sigma,train_data,center,train_label,lambda,sigma_flag);
     sigma = 10000;
    [TePred,error] = Q2T2RBFN_test(sigma,test_data,center,weights,test_label);
    
    TrAcc = zeros(1,1000);
    TeAcc = zeros(1,1000);
    thr = zeros(1,1000);
    TrLabel = train_label;
    TeLabel = test_label;
    TrN = length(TrLabel);
    TeN = length(TeLabel);
    for i = 1:1000
        t = (max(TrPred)-min(TrPred)) * (i-1)/1000 + min(TrPred);
        thr(i) = t;
        TrAcc(i) = (sum(TrLabel(TrPred<t)==0) + sum(TrLabel(TrPred>=t)==1)) / TrN;
        TeAcc(i) = (sum(TeLabel(TePred<t)==0) + sum(TeLabel(TePred>=t)==1)) / TeN;
    end
    plot(thr,TrAcc,'.- ',thr,TeAcc,'^-');legend('tr','te');
    title('Sigma :',num2str(sigma));



elseif question == 3
    center = train_data(:,randperm(600,2));
    c= Kmeans(train_data,center);
    
    train_idx_0 = find(train_label==0);
    K_data_0 = train_data(:,train_idx_0);
    
    train_idx_1 = find(train_label==1);
    K_data_1 = train_data(:,train_idx_1);
    
    train_idx_2 = find(train_label==2);
    K_data_2 = train_data(:,train_idx_2);
    
    train_idx_3 = find(train_label==3);
    K_data_3 = train_data(:,train_idx_3);
    
    train_idx_4 = find(train_label==4);
    K_data_4 = train_data(:,train_idx_4);
    
    train_idx_5 = find(train_label==5);
    K_data_5 = train_data(:,train_idx_5);
    
    train_idx_6 = find(train_label==6);
    K_data_6 = train_data(:,train_idx_6);
    
    train_idx_7 = find(train_label==7);
    K_data_7 = train_data(:,train_idx_7);
    
    train_idx_8 = find(train_label==8);
    K_data_8 = train_data(:,train_idx_8);
    
    train_idx_9 = find(train_label==9);
    K_data_9 = train_data(:,train_idx_9);
    
    norm1 = max(c(:,1),[],'all') - min(c(:,1),[],'all');
    norm2 = max(c(:,2),[],'all') - min(c(:,2),[],'all');
    c(:,1) = c(:,1)/norm1;
    c(:,2) = c(:,2)/norm2;
%     subplot(2,1,1);
%     imshow(reshape(c(1,:),28,28));
%     title('Cluster Centers');
%     subplot(2,1,2);
%     imshow(reshape(c(2,:),28,28));
%     title('Cluster Centers');
    
    sigma = 913;
    lambda = 0;
    sigma_flag = 1;
    
    [sigma,TrPred,weights] = Q2RBFN_train(sigma,train_data,center,train_label,lambda,sigma_flag);
     sigma = 913;
    [TePred,error] = Q2T2RBFN_test(sigma,test_data,center,weights,test_label);
    
    
    TrAcc = zeros(1,1000);
    TeAcc = zeros(1,1000);
    thr = zeros(1,1000);
    TrLabel = train_label;
    TeLabel = test_label;
    TrN = length(TrLabel);
    TeN = length(TeLabel);
    for i = 1:1000
        t = (max(TrPred)-min(TrPred)) * (i-1)/1000 + min(TrPred);
        thr(i) = t;
        TrAcc(i) = (sum(TrLabel(TrPred<t)==0) + sum(TrLabel(TrPred>=t)==1)) / TrN;
        TeAcc(i) = (sum(TeLabel(TePred<t)==0) + sum(TeLabel(TePred>=t)==1)) / TeN;
    end
    plot(thr,TrAcc,'.- ',thr,TeAcc,'^-');legend('tr','te');
    title('K-means Clustering,Sigma :',num2str(sigma));
end

%% Q3 A
clear; clc; close all;
x = linspace(-4*pi,4*pi,400);
trainX = [x; sin(x)./x]; % 2x400 matrix
plot(trainX(1,:),trainX(2,:),'r');% axis equal
hold on;
w = (2*rand(40,1)-1)*pi;
sigma_0 = 10;
iter = 500;
tao1 = iter/log(sigma_0);% time decay constant for h
tao2 = 1000;% time decay constant for learning rate
sequence = 1:40;
sequence = sequence';
t = 1;%t: iteration 0,1,2....


sigma = sigma_0 * exp(-(t-1)/tao1);
lr0 = 1;
lr = lr0 * exp(- (t-1)/tao2);


for t = 1: iter
    sample = x(randperm(400,1));
    dis = sqrt((w - sample).^2);
    idx = find(dis == min(dis));
    idx = idx(1);
    sigma = sigma_0 * exp(-(t-1)/tao1);
    if lr>0.01
        lr = lr0 * exp(- (t-1)/tao2);
    end
    dis_center = sqrt((sequence - sequence(idx)).^2);
    h = exp(- dis_center/(2*sigma^2));
    w = w + lr * h .* (sample - w);
    plot(w, sin(w)./w,'r');
end
b = plot(w,  sin(w)./w, 'b');
hold on;
a = plot(x,sin(x)./x,'r');
legend(a,'Ground Truth');
hold off;
b = plot(w, sin(w)./w, 'b');
hold on;
N = 40;    
for i = 1:N 
    center = w(i);
    if i == 1
        neighbor_1 = nan;
    else
        neighbor_1 = w(i-1);
    end
    if i == N
        neighbor_2 = nan;
    else 
        neighbor_2 = w(i+1);
    end
    if isnan(neighbor_1)
    else
        plot([center,neighbor_1],[sin(center)./center,sin(center)./center],'g','LineWidth',3);
    end
    if isnan(neighbor_2)
    else
        plot([center,neighbor_2],[sin(center)./center,sin(center)./center],'g','LineWidth',3);
    end
end

   
%% Q3 B
clear; clc; close all;
X = randn(800,2);
s2 = sum(X.^2,2);

trainX = (X.*repmat(1*(gammainc(s2/2,1).^(1/2))./sqrt(s2),1,2))';
plot(trainX(1,:),trainX(2,:),'+r');
hold on;
N = 8; 
w = (2*rand(2,N,N)-1)/10;
sigma0 = 200;
tao1 = 1000/log(sigma0);
tao2 = 1000;
dis_center = zeros(N,N);
t = 0;
sigma = sigma0 * exp(-(t-1)/tao1);
lr0 = 1;
lr = lr0 * exp(- t/tao2);
iter = 5000;
for t = 1:iter
    sample = trainX(:,randperm(800,1));
    dis = sqrt(sum((w - sample).^2));
    dis = reshape(dis,N,N);
    [p,q] = find(dis == min(dis,[],'all'));
    p = p(1);
    q = q(1);
    if sigma >0.5
        sigma = sigma0 * exp(-(t-1)/tao1);
    end
    if lr>0.01
        lr = lr0 * exp(- (t-1)/tao2);
    end
    for i = 1:N
        for k = 1:N
            dis_center(i,k) = sqrt((i-p)^2 + (k-q)^2);
        end
    end
    h = exp(- dis_center/(2*sigma^2));
    h = reshape(h,1,N,N);
    w = w + lr * h .* (sample - w);
end
plot(w(1,:),w(2,:),'+b','LineWidth',3);

for i = 1:N
    for k = 1:N
        center = [w(1,i,k),w(2,i,k)];
        if i == 1
            neiborU = [nan,nan];
        else
            neiborU = [w(1,i-1,k),w(2,i-1,k)];
        end
        if i == N
            neiborD = [nan,nan];
        else
            neiborD = [w(1,i+1,k),w(2,i+1,k)];
        end
        if k == 1
            neiborL = [nan,nan];
        else
            neiborL = [w(1,i,k-1),w(2,i,k-1)];
        end
        if k== N
            neiborR = [nan,nan];
        else
            neiborR = [w(1,i,k+1),w(2,i,k+1)];
        end
        neibor = [neiborU;neiborD;neiborL;neiborR];
        for j = 1:4
            if isnan(neibor(j,1))
            else
                plot([center(1),neibor(j,1)],[center(2),neibor(j,2)],'b','LineWidth',2);
            end
        end
    end
end
title('Iteration:',num2str(iter));

%% Q3 C
% last digit:6   mod(6,5)=1, mod(7,5)=2, omit class 1 and 2
% choose class 0,3,4
clear; clc; close all;
load Digits.mat
% train data
trainIdx_0 = find(train_classlabel==0);
trainX_0 = train_data(:,trainIdx_0);
trainY_0 = train_classlabel(trainIdx_0);
trainY_0(trainY_0(:,:)==0) = 0;

trainIdx_3 = find(train_classlabel==3);
trainX_3 = train_data(:,trainIdx_3);
trainY_3 = train_classlabel(trainIdx_3);
trainY_3(trainY_3(:,:)==3) = 3;

trainIdx_4 = find(train_classlabel==4);
trainX_4 = train_data(:,trainIdx_4);
trainY_4 = train_classlabel(trainIdx_4);
trainY_4(trainY_4(:,:)==4) = 4;

% test data 
testIdx_0 = find(test_classlabel==0);
testX_0 = test_data(:,testIdx_0);
testY_0 = test_classlabel(testIdx_0);
testY_0(testY_0(:,:)==0) = 0;

testIdx_3 = find(test_classlabel==3);
testX_3 = test_data(:,testIdx_3);
testY_3 = test_classlabel(testIdx_3);
testY_3(testY_3(:,:)==3) = 3;

testIdx_4 = find(test_classlabel==4);
testX_4 = test_data(:,testIdx_4);
testY_4 = test_classlabel(testIdx_4);
testY_4(testY_4(:,:)==4) = 4;
% Combine
trainX = [trainX_0 trainX_3 trainX_4]';
trainY = [trainY_0  trainY_3  trainY_4];
testX = [testX_0 testX_3 testX_4]';
testY = [testY_0  testY_3  testY_4];

trainX = double(trainX);
trainY = double(trainY);
testX = double(testX);
testY = double(testY);

N = 10;   
w = 2*rand(784,N,N)-1;
sigma0 = 500;
tao1 = 1000/log(sigma0);
tao2 = 600;
dis_center = zeros(N,N);
t = 0;
sigma = sigma0 * exp(-(t-1)/tao1);
lr0 = 0.5;
lr = lr0 * exp(- t/tao2);
iter = 1000;
for t = 1:iter
    sample = double(trainX(randperm(600,1),:)');
    dis = sqrt(sum((w - sample).^2,1));
    dis = reshape(dis,N,N);
    [p,q] = find(dis == min(dis,[],'all'));
    if sigma >0.5
        sigma = sigma0 * exp(-(t-1)/tao1);
    end
    if lr > 0.01
        lr = lr0 * exp(- (t-1)/tao2);
    end
    for i = 1:N
        for k = 1:N
            dis_center(i,k) = sqrt((i-p)^2 + (k-q)^2);
        end
    end
    h = exp(- dis_center/(2*sigma^2));
    h = reshape(h,1,N,N);
    w = w + lr * h .* (sample - w);
end


skip = 0;
if skip == 0
    for i = 1: N
        for k = 1:N
            w1 = reshape(w(:,i,k),28,28);
            norm1 = max(w1,[],'all');
            norm2 = min(w1,[],'all');
            norm = norm1 - norm2;
            w1 = w1/norm;
            subplot(10,10,(i-1)*10 + k);
            imshow(w1);
        end
    end
end
close all;
dis_w = zeros(600,1);
Label = zeros(N,N);
for i = 1:10
    for k = 1:10
        for j = 1: 600
            dis_w(j) = sqrt(sum((double(trainX(j,:)') - w(:,i,k)).^2,'all'));
        end
        [p,q] = find(dis_w == min(dis_w,[],'all'));
        Label(i,k) = trainY(p(1));   % <---
        if Label(i,k)==0
            plot(i,k,'bo','MarkerFaceColor','b','Linewidth',3);
            hold on
        else
            plot(i,k,'ro','MarkerFaceColor','r','Linewidth',3);
            hold on;
        end
    end
end
Ax=gca;
Ax.XAxisLocation='top';
Ax.YDir='reverse';
dis_test = zeros(N,N);
TePred = zeros(1,60);

for j = 1:60
    for i = 1:10
        for k = 1:10
            dis_test(i,k) = sqrt(sum((double(testX(j,:)') - w(:,i,k)).^2,'all'));
        end
    end
    [p,q] = find(dis_test == min(dis_test,[],'all'));
    TePred(j) = Label(p,q);
end
fprintf('Testing accuracy is %6.5f%% \n',(1 - sum((abs(TePred - testY)))/100)*100);









