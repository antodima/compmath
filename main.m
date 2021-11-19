clear variables;

%% load datasets
monks1_train_filename = 'datasets/monks/monks-1.train';
monks2_train_filename = 'datasets/monks/monks-2.train';
monks3_train_filename = 'datasets/monks/monks-3.train';
monks1_test_filename = 'datasets/monks/monks-1.test';
monks2_test_filename = 'datasets/monks/monks-2.test';
monks3_test_filename = 'datasets/monks/monks-3.test';
cup_filename = 'datasets/cup/ml-cup.csv';

monks1_train = readtable(monks1_train_filename,'FileType','text');
monks1_x_train = table2array(monks1_train(:,1:6));
monks1_y_train = table2array(monks1_train(:,7));

monks1_test = readtable(monks1_test_filename,'FileType','text');
monks1_x_test = table2array(monks1_test(:,1:6));
monks1_y_test = table2array(monks1_test(:,7));

monks2_train = readtable(monks2_train_filename,'FileType','text');
monks2_x_train = table2array(monks2_train(:,1:6));
monks2_y_train = table2array(monks2_train(:,7));

monks2_test = readtable(monks2_test_filename,'FileType','text');
monks2_x_test = table2array(monks2_test(:,1:6));
monks2_y_test = table2array(monks2_test(:,7));

monks3_train = readtable(monks3_train_filename,'FileType','text');
monks3_x_train = table2array(monks3_train(:,1:6));
monks3_y_train = table2array(monks3_train(:,7));

monks3_test = readtable(monks3_test_filename,'FileType','text');
monks3_x_test = table2array(monks3_test(:,1:6));
monks3_y_test = table2array(monks3_test(:,7));

cup_train = readtable(cup_filename,'FileType','text');
cup_x_train = table2array(cup_train(1:1300,1:20));
cup_y_train = table2array(cup_train(1:1300,21:22));

cup_x_test = table2array(cup_train(1301:end,1:20));
cup_y_test = table2array(cup_train(1301:end,21:22));

%% the problem
%{
A = [2 5;1 7];
b = [100 70]';

x1 = -150; x2 = 150; interval = x1:5:x2;
%x0 = [-37,88]';
x0 = randi([x1,x2], size(b,1),1);

t = 0.01;
eps = 1e-6;
MaxIter = 1000;

[Problem] = quadratic(A, b, interval);
%}

h = 3; % hidden layer dimension
X = monks1_x_train; y = monks1_y_train;
W = rand(size(X,2),h);
Q = sigmoid(X*W);
A = Q'*Q; b = Q'*y; % non square matrix, solve: Q^T*Q=Q^T*b

x1 = -20; x2 = 20;
x0 = (x2-x1).*rand(size(b,1), 1, 'double');

t = 0.000001; eps = 1e-6; MaxIter = 100; l = 1e-4;
[Problem] = leastsquares(A, b, l);

%% the solutions
[x] = GD(Problem, x0, eps, t, MaxIter);




%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = sigmoid(a)
    y = 1.0 ./ (1 + exp(-a));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = relu(a)
    y = max(0, a);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = linear(x)
   y = x;
end