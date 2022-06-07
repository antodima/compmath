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
x0 = [-37,88]';
%x0 = randi([x1,x2], size(b,1),1);
lr = 0.1; eps = 1e-6; MaxIter = 1000; beta = 0.1; m1 = 0.0001; tau = 0.85; l = 1e-4;
[Problem] = quadratic(A, b, interval);
%[Problem] = leastsquares(A, b, 1e-5);
I = eye(size(A,2)); AA = A'*A+l*I; bb = A'*b;


%{
%X = monks1_x_train; y = monks1_y_train;
X = cup_x_train; y = cup_y_train;
X_test = cup_x_test; y_test = cup_y_test;

lr = 0.01; eps = 1e-8; MaxIter = 1000; l = 1e-4; beta = 0.01;
h = 3; m1 = 0.0001; tau = 0.9;
[Problem] = extreme(X, y, X_test, y_test, "sigmoid", h, l, false);
A = Problem.A; b = Problem.b; x0 = Problem.W2;
I = eye(size(A,2)); AA = A'*A+l*I; bb = A'*b;
%}

%{
lr = 0.01; eps = 1e-8; MaxIter = 1000; l = 1e-4; beta = 0.01;
h = 3; m1 = 0.0001; tau = 0.9;
A = randn(6,4); b = randn(6,1); x0 = randn(4,1);
[Problem] = leastsquares(A, b, 1e-5);
I = eye(size(A,2)); AA = A'*A+l*I; bb = A'*b;
%}

%% the solutions
[y1, iters1] = GD(Problem, x0, eps, lr, m1, tau, MaxIter, 'black', '-');
[y2, iters2] = HB(Problem, x0, eps, lr, beta, MaxIter, 'red', '-');
%[y3, iters3] = ACG(Problem, x0, eps, lr, beta, MaxIter, false, 'green', '-');
%[y4, iters4] = ADAM(Problem, x0, eps, 4, beta, 500, false, 'blue', '-');
[y5, iters5] = ADAM(Problem, x0, eps, 4, 0.9, MaxIter, true, 'yellow', '-');
[y6, iters8, etr8, ets8] = FISTA(Problem, x0, eps, MaxIter, 'blue', '-', 1);

% https://www.mit.edu/~9.520/spring10/Classes/class04-rls.pdf
[L7,D7] = ldl(AA); y7 = L7' \ ((L7\bb) ./ diag(D7));
[L8, D8, P8, y8] = LDL(AA, bb, true);
[L9, D9, P9, y9] = LDL(AA, bb, false);

format short e;
disp("======================================================================");
e1 = sqrt(immse(b, A*y1)); r1 = norm(b-A*y1)/norm(b); fprintf('GD \t (black): \t\t iters=%d \t rmse=%e \t residual=%e \n', iters1, e1, r1);
e2 = sqrt(immse(b, A*y2)); r2 = norm(b-A*y2)/norm(b); fprintf('HB \t (red): \t\t iters=%d \t rmse=%e \t residual=%e \n', iters2, e2, r2);
%e3 = sqrt(immse(b, A*y3)); r3 = norm(b-A*y3)/norm(b); fprintf('ACG \t (green): \t\t iters=%d \t rmse=%e \t residual=%e \n', iters3, e3, r3);
%e4 = sqrt(immse(b, A*y4)); r4 = norm(b-A*y4)/norm(b); fprintf('ADAM \t (blue): \t\t iters=%d \t rmse=%e \t residual=%e \n', iters4, e4, r4);
e5 = sqrt(immse(b, A*y5)); r5 = norm(b-A*y5)/norm(b); fprintf('NADAM \t (yellow): \t\t iters=%d \t rmse=%e \t residual=%e \n', iters5, e5, r5);
e6 = sqrt(immse(b, A*y6)); r6 = norm(b-A*y6)/norm(b); fprintf('FISTA \t (blue): \t\t iters=%d \t rmse=%e \t residual=%e \n', iters8, e6, r6);

r7 = norm(bb-AA*y7)/norm(bb); fprintf('LDL \t (matlab): \t\t ----- \t\t residual=%e \t ∥A∥=%f ∥L∥=%f ∥D∥=%f\n', r7, norm(AA), norm(L7), norm(D7));
r8 = norm(bb-AA*y8)/norm(bb); fprintf('LDL \t (with pivoting): \t ----- \t\t residual=%e \t ∥A∥=%f ∥L∥=%f ∥D∥=%f\n', r8, norm(AA), norm(L8), norm(D8));
r9 = norm(bb-AA*y9)/norm(bb); fprintf('LDL \t (no pivoting): \t ----- \t\t residual=%e \t ∥A∥=%f ∥L∥=%f ∥D∥=%f\n', r9, norm(AA), norm(L9), norm(D9));

if Problem.name == "quadratic"
    [h,icons,plots,legend_text] = Problem.plot_legend('GD','HB','NADAM','FISTA');
    
end


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
function g = gridSearch(hiddenSizes,epochs,learningRates,lambdas)
    sets = {hiddenSizes,epochs,learningRates,lambdas};
    [H,E,LR,L] = ndgrid(sets{:});
    g = [H(:) E(:) LR(:) L(:)];
end