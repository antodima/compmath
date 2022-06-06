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
X = cup_x_train; y = cup_y_train;
X_test = cup_x_test; y_test = cup_y_test;

format short e;

rmdir('results','s')
mkdir('results')

% hyperparameters
hiddenSizes = 100:100:500;
epochs = 100:200:1000;
learningRates = [0.001,0.01,0.1];
lambdas = [0.0001,0.001,0.01,0.1];
grid = gridSearch(hiddenSizes, epochs, learningRates, lambdas);
rmses = []; residuals = [];
for g=1:size(grid,1)
    params = grid(g,:);
    h = params(1);
    MaxIter = params(2);
    lr = params(3);
    l = params(4);
    eps = 1e-8; tau = 0.9;
    fprintf('%d: h=%3d, MaxIter=%4d, lr=%1.4e, lambda=%1.4e \n', g, h, MaxIter, lr, l);
    
    [Problem] = extreme(X, y, X_test, y_test, "sigmoid", h, l, false);
    A = Problem.A; b = Problem.b; x0 = Problem.W2;
    
    [x, iters, errors_train, errors_test] = FISTA(Problem, x0, eps, MaxIter, 'blue', '-', 0);
    
    e = sqrt(immse(b, A*x)); r = norm(b-A*x)/norm(b);
    fprintf('\t iterations=%d \t rmse=%e \t residual=%e \n', iters, e, r);
    
    rmses(end+1) = e;
    residuals(end+1) = r;
    save(sprintf('results/x%d.mat',g),'x');
    save(sprintf('results/errors_train%d.mat',g),'errors_train');
    save(sprintf('results/errors_test%d.mat',g),'errors_test');
end
save('results/rmse.mat','rmses');
save('results/residual.mat','residuals');
save('results/grid.mat','grid');

% best result
disp("Best result:");
load('results/residual.mat'); load('results/grid.mat'); 
[value,pos] = min(residuals); fprintf('h=%d, epochs=%d, lr=%1.4e, lambda=%1.4e, residual=%1.4e \n', grid(pos,:), value); 
load(sprintf('results/x%d.mat',pos));
load(sprintf('results/errors_train%d.mat',pos));
load(sprintf('results/errors_test%d.mat',pos));



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function g = gridSearch(hiddenSizes,epochs,learningRates,lambdas)
    sets = {hiddenSizes,epochs,learningRates,lambdas};
    [H,E,LR,L] = ndgrid(sets{:});
    g = [H(:) E(:) LR(:) L(:)];
end