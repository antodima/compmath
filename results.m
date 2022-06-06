clear variables;

%% load datasets
cup_filename = 'datasets/cup/ml-cup.csv';

cup_train = readtable(cup_filename,'FileType','text');
cup_x_train = table2array(cup_train(1:1300,1:20));
cup_y_train = table2array(cup_train(1:1300,21:22));
cup_x_test = table2array(cup_train(1301:end,1:20));
cup_y_test = table2array(cup_train(1301:end,21:22));

%% results
load('results/residual.mat'); load('results/grid.mat'); 
[value,pos] = min(residuals); fprintf('Best model: h=%d, epochs=%d, lr=%1.4e, lambda=%1.4e, residual=%1.4e \n', grid(pos,:), value); 
load(sprintf('results/x%d.mat',pos));
load(sprintf('results/errors_train%d.mat',pos));
load(sprintf('results/errors_test%d.mat',pos));

plot(errors_train,'-','LineWidth',1);
hold on;
plot(errors_test,'-','LineWidth',1);
xlabel('epochs')
ylabel('loss');
title('Best model learning curves');
legend('training set','test set');


A = cup_x_train; b = cup_y_train; I = eye(size(A,2)); l = 1e-4;
AA = A'*A+l*I; bb = A'*b;

A_test = cup_x_test; b_test = cup_y_test;
AA_test = A_test'*A_test+l*I; bb_test = A_test'*b_test;

[L1, D1, P1, y1] = LDL(AA, bb, false);
[L2, D2, P2, y2] = LDL(AA, bb, true);

res_train_1 = norm(bb-AA*y1)/norm(bb); res_test_1 = norm(bb_test-AA_test*y1)/norm(bb_test);
res_train_2 = norm(bb-AA*y2)/norm(bb); res_test_2 = norm(bb_test-AA_test*y2)/norm(bb_test);

fprintf('FISTA \t\t\t residual (train)=%1.4e \t residual (test)=%1.4e \n', value, errors_test(end));
fprintf('LDL (no pivot) \t\t residual (train)=%1.4e \t residual (test)=%1.4e \n', res_train_1, res_test_1);
fprintf('LDL (with pivot) \t residual (train)=%1.4e \t residual (test)=%1.4e \n', res_train_2, res_test_2);

