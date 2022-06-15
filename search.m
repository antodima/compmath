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

format short e;

rmdir('results','s')
mkdir('results')

% hyperparameters
hiddenSizes = 10:30:100;
epochs = 3000;
learningRates = [0.001, 0.01, 0.1];
lambdas = [0.001, 0.01, 0.1];
grid_fista = gridSearch(hiddenSizes, epochs, learningRates, lambdas);
grid_losses_fista = [];
disp("Grid search FISTA:");
for g=1:size(grid_fista,1)
    params = grid_fista(g,:);
    h = params(1);
    MaxIter = params(2);
    lr = params(3);
    l = params(4);
    m = 0.9;
    eps = 10;
    fprintf('%d: h=%3d, MaxIter=%4d, lr=%1.4e, lambda=%1.4e\n', g, h, MaxIter, lr, l);
    
    [Problem] = extreme(X, y, "sigmoid", h, l, false);
    A = Problem.A; b = Problem.b; x0 = Problem.W2;
    
    tic; [x, iters, losses, norms] = FISTA(Problem, x0, eps, MaxIter, 'blue', '-', 0);
    elapsed_time=toc;
    fprintf('\t iteration=%d \t loss=%e \n', iters, losses(end));
    
    grid_losses_fista(end+1) = losses(end);
    save(sprintf('results/x%d.mat',g),'x');
    save(sprintf('results/losses%d.mat',g),'losses');
    save(sprintf('results/norms%d.mat',g),'norms');
    save(sprintf('results/elapsed_time%d.mat',g),'elapsed_time');
    residual = norm(b-A*x)/norm(b); save(sprintf('results/residual%d.mat',g),'residual');
end
save('results/grid_losses_fista.mat','grid_losses_fista')
save('results/grid_fista.mat','grid_fista');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% hyperparameters for grid_gd
hiddenSizes = 10:30:100;
epochs = 1000;
learningRates = [0.001,0.01];
lambdas = [0.001,0.01];
grid_gd = gridSearch(hiddenSizes, epochs, learningRates, lambdas);
grid_losses_gd = [];
disp("Grid search GD:");
for g=1:size(grid_gd,1)
    params = grid_gd(g,:);
    h = params(1);
    MaxIter = params(2);
    lr = params(3);
    l = params(4);
    m = 0.9;
    tau = 0;
    eps = 20;
    fprintf('%d: h=%3d, MaxIter=%4d, lr=%1.4e, lambda=%1.4e\n', g, h, MaxIter, lr, l);
    
    [Problem] = extreme(X, y, "sigmoid", h, l, false);
    A = Problem.A; b = Problem.b; x0 = Problem.W2;
    
    tic; [x_gd, iters_gd, losses_gd, norms_gd] = GD(Problem, x0, eps, lr, m, tau, MaxIter, 'red', '-', 0);
    elapsed_time_gd=toc;
    fprintf('\t iterations=%d \t loss=%e \n', iters_gd, losses_gd(end));

    grid_losses_gd(end+1) = losses_gd(end);
    save(sprintf('results/x_gd%d.mat',g),'x_gd');
    save(sprintf('results/losses_gd%d.mat',g),'losses_gd');
    save(sprintf('results/norms_gd%d.mat',g),'norms_gd');
    save(sprintf('results/elapsed_time_gd%d.mat',g),'elapsed_time_gd');
    residual_gd = norm(b-A*x_gd)/norm(b); save(sprintf('results/residual_gd%d.mat',g),'residual_gd');
    
end
save('results/grid_losses_gd.mat','grid_losses_gd')
save('results/grid_gd.mat','grid_gd');


% best result
disp("Best result (FISTA):");
load('results/grid_losses_fista.mat'); load('results/grid_fista.mat');
[value, pos] = min(grid_losses_fista); fprintf('h=%d, epochs=%d, lr=%1.4e, lambda=%1.4e, loss=%1.4e \n', grid_fista(pos,:), value); 

disp("Best result (GD):");
load('results/grid_losses_gd.mat');load('results/grid_gd.mat'); 
[value_gd, pos_gd] = min(grid_losses_gd); fprintf('h=%d, epochs=%d, lr=%1.4e, lambda=%1.4e, loss=%1.4e \n', grid_gd(pos_gd,:), value_gd); 


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function g = gridSearch(hiddenSizes,epochs,learningRates,lambdas)
    sets = {hiddenSizes,epochs,learningRates,lambdas};
    [H,E,LR,L] = ndgrid(sets{:});
    g = [H(:) E(:) LR(:) L(:)];
end