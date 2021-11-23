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
%{%}
A = [2 5;1 7];
b = [100 70]';

x1 = -150; x2 = 150; interval = x1:5:x2;
%x0 = [-37,88]';
x0 = randi([x1,x2], size(b,1),1);
lr = 0.1; eps = 1e-6; MaxIter = 100; beta = 0.1;
[Problem] = quadratic(A, b, interval);


%{
h = 3; % hidden layer dimension
X = monks1_x_train; y = monks1_y_train;
W = rand(size(X,2),h);
Q = sigmoid(X*W);
A = Q'*Q; b = Q'*y; % non square matrix, solve: Q^T*Q=Q^T*b

x1 = -20; x2 = 20;
x0 = (x2-x1).*rand(size(b,1), 1, 'double');

lr = 0.000001; eps = 1e-6; MaxIter = 100; l = 1e-4; beta = 0.01;
[Problem] = leastsquares(A, b, l);
%}

%% the solutions
[y1, iters1] = GD(Problem, x0, eps, lr, MaxIter, 'black', '-');
[y2, iters2] = HB(Problem, x0, eps, lr, beta, MaxIter, 'red', '-');
[y3, iters3] = ACG(Problem, x0, eps, lr, beta, MaxIter, false, 'green', '-');
[y4, iters4] = ADAM(Problem, x0, eps, 4, beta, 1000, false, 'blue', '-');
[y5, iters5] = ADAM(Problem, x0, eps, 4, 0.9, 1000, true, 'yellow', '-');

disp("===================");
fprintf('GD (black, -):\t\t iters=%d \t residual=%e\n', iters1, norm(b-A*y1));
fprintf('HB (red, -):\t\t iters=%d \t residual=%e\n', iters2, norm(b-A*y2));
fprintf('ACG (green, -):\t\t iters=%d \t residual=%e\n', iters3, norm(b-A*y3));
fprintf('ADAM (blue, -):\t\t iters=%d \t residual=%e\n', iters4, norm(b-A*y4));
fprintf('ADAM-acc (yellow, -):\t iters=%d \t residual=%e\n', iters5, norm(b-A*y5));

%{
% hyperparameters
hiddenSizes = 10:10:50;
epochs = 50:20:100;
learningRates = [0.001,0.01,0.1];
lambdas = [0.001,0.01,0.1];
grid = gridSearch(hiddenSizes,epochs,learningRates,lambdas);
for g=1:size(grid,1)
    params = grid(g,:);
    h = params(1);
    MaxIter = params(2);
    t = params(3);
    l = params(4);
end
%}



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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function g = gridSearch(hiddenSizes,epochs,learningRates,lambdas)
    sets = {hiddenSizes,epochs,learningRates,lambdas};
    [H,E,LR,L] = ndgrid(sets{:});
    g = [H(:) E(:) LR(:) L(:)];
end