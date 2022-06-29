clear
clear variables; format short e;
rmdir('results','s'); mkdir('results');

%% load datasets
cup_filename = 'datasets/cup/ml-cup.csv';

cup_train = readtable(cup_filename,'FileType','text');
cup_x_train = table2array(cup_train(1:1300,1:20));
cup_y_train = table2array(cup_train(1:1300,21:22));

%% problem setup
h = 100; l = 0.1;

X = cup_x_train; y = cup_y_train;
[Problem] = extreme(X, y, "sigmoid", h, l, false);
A = Problem.A; b = Problem.b; x0 = Problem.W2;
I = eye(size(A,2)); AA = A'*A+l*I; bb = A'*b;

[L,D] = ldl(AA);
xstar = L' \ ((L\bb) ./ diag(D)); fstar = Problem.cost(xstar);
save('results/xstar.mat','xstar');
save('results/fstar.mat','fstar');
save('results/Problem.mat','Problem');

%% search algorithm parameters
eps = 1e-4;
epochs = 20000;
learningRates = 1e-4;

grid_fista = gridSearch(h, epochs, learningRates, l);
grid_losses_fista = [];
disp("Grid search FISTA:");
for g=1:size(grid_fista,1)
    params = grid_fista(g,:);
    h = params(1);
    MaxIter = params(2);
    lr = params(3);
    l = params(4);
    
    tic; [x, iters, losses, norms] = FISTA(Problem, x0, fstar, eps, MaxIter, 'blue', '-', 0);
    elapsed_time=toc;
    fprintf('%d: MaxIter=%4d, eps=%1.4e, iterations=%d, loss=%e, f*=%e \n', g, MaxIter, eps, iters, losses(end), fstar);
    
    grid_losses_fista(end+1) = losses(end);
    save(sprintf('results/x%d.mat',g),'x');
    save(sprintf('results/losses%d.mat',g),'losses');
    save(sprintf('results/norms%d.mat',g),'norms');
    save(sprintf('results/elapsed_time%d.mat',g),'elapsed_time');
    AA = A'*A; bb = A'*b; residual = norm(bb-AA*x)/norm(bb); save(sprintf('results/residual%d.mat',g),'residual');
end
save('results/grid_losses_fista.mat','grid_losses_fista')
save('results/grid_fista.mat','grid_fista');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% hyperparameters for grid_gd
grid_gd = gridSearch(h, epochs, learningRates, l);
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
    
    tic; [x_gd, iters_gd, losses_gd, norms_gd] = GD(Problem, x0, fstar, eps, lr, m, tau, MaxIter, 'red', '-', 0);
    elapsed_time_gd=toc;
    fprintf('%d: MaxIter=%4d, eps=%1.4e, lr=%1.4e, iterations=%d, loss=%e, f*=%e \n', g, MaxIter, eps, lr, iters_gd, losses_gd(end), fstar);

    grid_losses_gd(end+1) = losses_gd(end);
    save(sprintf('results/x_gd%d.mat',g),'x_gd');
    save(sprintf('results/losses_gd%d.mat',g),'losses_gd');
    save(sprintf('results/norms_gd%d.mat',g),'norms_gd');
    save(sprintf('results/elapsed_time_gd%d.mat',g),'elapsed_time_gd');
    AA = A'*A; bb = A'*b; residual_gd = norm(bb-AA*x)/norm(bb); save(sprintf('results/residual_gd%d.mat',g),'residual_gd');
end
save('results/grid_losses_gd.mat','grid_losses_gd')
save('results/grid_gd.mat','grid_gd');


% best result
fprintf('\nProblem parameters: hidden size=%3d, lambda=%1.4e \n', Problem.h, Problem.l);

load('results/grid_losses_fista.mat'); load('results/grid_fista.mat');
[value, pos] = min(grid_losses_fista);
params = grid_fista(pos,:); h = params(1); MaxIter = params(2); lr = params(3); l = params(4);
fprintf('FISTA best result: iterations=%d, loss=%e, f*=%e \n', MaxIter, value, fstar);

load('results/grid_losses_gd.mat'); load('results/grid_gd.mat'); 
[value_gd, pos_gd] = min(grid_losses_gd);
params_gd = grid_gd(pos_gd,:); h = params_gd(1); MaxIter = params_gd(2); lr = params_gd(3); l = params_gd(4);
fprintf('GD best result: iterations=%d, lr=%e loss=%e, f*=%e \n', MaxIter, lr, value_gd, fstar);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function g = gridSearch(hiddenSizes,epochs,learningRates,lambdas)
    sets = {hiddenSizes,epochs,learningRates,lambdas};
    [H,E,LR,L] = ndgrid(sets{:});
    g = [H(:) E(:) LR(:) L(:)];
end