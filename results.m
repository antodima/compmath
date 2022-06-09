clear variables;

%% load datasets
cup_filename = 'datasets/cup/ml-cup.csv';

cup_train = readtable(cup_filename,'FileType','text');
cup_x_train = table2array(cup_train(1:1300,1:20));
cup_y_train = table2array(cup_train(1:1300,21:22));
cup_x_test = table2array(cup_train(1301:end,1:20));
cup_y_test = table2array(cup_train(1301:end,21:22));

%% results

load('results/residuals.mat'); load('results/grid.mat'); 
[value,pos] = min(residuals); fprintf('Best model: h=%d, epochs=%d, lr=%1.4e, lambda=%1.4e, residual=%1.4e \n', grid(pos,:), value); 

load(sprintf('results/x%d.mat',pos),'x');
load(sprintf('results/loss%d.mat',pos),'loss');
load(sprintf('results/loss_test%d.mat',pos),'loss_test');
load(sprintf('results/errors%d.mat',pos),'errors');
load(sprintf('results/errors_test%d.mat',pos),'errors_test');
load(sprintf('results/rates%d.mat',pos),'rates');
load(sprintf('results/norms%d.mat',pos),'norms');

figure();
plot(loss,'-','LineWidth',1);
hold on;
plot(loss_test,'-','LineWidth',1);
xlabel('Epochs')
ylabel('MSE');
legend('training set','test set');
hold off;

figure();
plot(errors,'-','LineWidth',1);
hold on;
plot(errors_test,'-','LineWidth',1);
xlabel('Epochs')
ylabel('Residual');
legend('training set','test set');
hold off;

figure();
plot(rates,'-','LineWidth',1);
xlabel('Epochs')
ylabel('Convergence rate');
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A = cup_x_train; b = cup_y_train; I = eye(size(A,2)); l = 1e-4;
AA = A'*A+l*I; bb = A'*b;

A_test = cup_x_test; b_test = cup_y_test;
AA_test = A_test'*A_test+l*I; bb_test = A_test'*b_test;

[L1, D1, P1, y1] = LDL(AA, bb, false);
[L2, D2, P2, y2] = LDL(AA, bb, true);
[L3,D3] = ldl(AA); y3 = L3' \ ((L3\bb) ./ diag(D3));

res_train_1 = norm(bb-AA*y1)/norm(bb); res_test_1 = norm(bb_test-AA_test*y1)/norm(bb_test);
res_train_2 = norm(bb-AA*y2)/norm(bb); res_test_2 = norm(bb_test-AA_test*y2)/norm(bb_test);
res_train_3 = norm(bb-AA*y3)/norm(bb); res_test_3 = norm(bb_test-AA_test*y3)/norm(bb_test);

hold off;
disp("======================================================================");
disp("CUP results:");
fprintf('FISTA \t\t\t residual (train)=%1.4e \t residual (test)=%1.4e \n', value, errors_test(end));
fprintf('LDL (no pivot) \t\t residual (train)=%1.4e \t residual (test)=%1.4e \n', res_train_1, res_test_1);
fprintf('LDL (with pivot) \t residual (train)=%1.4e \t residual (test)=%1.4e \n', res_train_2, res_test_2);
fprintf('LDL (matlab) \t\t residual (train)=%1.4e \t residual (test)=%1.4e \n', res_train_3, res_test_3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure();

A = [2 5;1 7];
b = [100 70]';

x1 = -150; x2 = 150; interval = x1:5:x2;
x0 = [-37,88]';
%x0 = randi([x1,x2], size(b,1),1);
lr = 0.1; eps = 1e-6; MaxIter = 1000; beta = 0.1; m1 = 0.0001; tau = 0.85; l = 1e-4;
[Problem] = quadratic(A, b, interval);
%[Problem] = leastsquares(A, b, 1e-5);
I = eye(size(A,2)); AA = A'*A+l*I; bb = A'*b;

[y1, iters1] = GD(Problem, x0, eps, lr, m1, tau, MaxIter, 'black', '-', 0);
[y2, iters2] = HB(Problem, x0, eps, lr, beta, MaxIter, 'red', '-', 0);
%[y3, iters3] = ACG(Problem, x0, eps, lr, beta, MaxIter, false, 'green', '-');
%[y4, iters4] = ADAM(Problem, x0, eps, 4, beta, 500, false, 'blue', '-');
[y5, iters5] = ADAM(Problem, x0, eps, 4, 0.9, MaxIter, true, 'blue', '-', 0);
[y6, iters6, loss6, loss_test6, errors6, errors_test6, rates6, norms6] = FISTA(Problem, x0, eps, MaxIter, 'green', '-', 0);
[L7,D7] = ldl(AA); y7 = L7' \ ((L7\bb) ./ diag(D7));
[L8, D8, P8, y8] = LDL(AA, bb, true);
[L9, D9, P9, y9] = LDL(AA, bb, false);

hold off;
disp("======================================================================");
disp("QUADRATIC results:");
format short e;
e1 = sqrt(immse(b, A*y1)); r1 = norm(b-A*y1)/norm(b); fprintf('GD \t (black): \t\t iters=%d \t rmse=%e \t residual=%e \n', iters1, e1, r1);
e2 = sqrt(immse(b, A*y2)); r2 = norm(b-A*y2)/norm(b); fprintf('HB \t (red): \t\t iters=%d \t rmse=%e \t residual=%e \n', iters2, e2, r2);
%e3 = sqrt(immse(b, A*y3)); r3 = norm(b-A*y3)/norm(b); fprintf('ACG \t (green): \t\t iters=%d \t rmse=%e \t residual=%e \n', iters3, e3, r3);
%e4 = sqrt(immse(b, A*y4)); r4 = norm(b-A*y4)/norm(b); fprintf('ADAM \t (blue): \t\t iters=%d \t rmse=%e \t residual=%e \n', iters4, e4, r4);
e5 = sqrt(immse(b, A*y5)); r5 = norm(b-A*y5)/norm(b); fprintf('NADAM \t (blue): \t\t iters=%d \t rmse=%e \t residual=%e \n', iters5, e5, r5);
e6 = sqrt(immse(b, A*y6)); r6 = norm(b-A*y6)/norm(b); fprintf('FISTA \t (green): \t\t iters=%d \t rmse=%e \t residual=%e \n', iters6, e6, r6);

r7 = norm(bb-AA*y7)/norm(bb); fprintf('LDL \t (matlab): \t\t ----- \t\t residual=%e \t ∥A∥=%f ∥L∥=%f ∥D∥=%f\n', r7, norm(AA), norm(L7), norm(D7));
r8 = norm(bb-AA*y8)/norm(bb); fprintf('LDL \t (with pivoting): \t ----- \t\t residual=%e \t ∥A∥=%f ∥L∥=%f ∥D∥=%f\n', r8, norm(AA), norm(L8), norm(D8));
r9 = norm(bb-AA*y9)/norm(bb); fprintf('LDL \t (no pivoting): \t ----- \t\t residual=%e \t ∥A∥=%f ∥L∥=%f ∥D∥=%f\n', r9, norm(AA), norm(L9), norm(D9));

if Problem.name == "quadratic"
    Problem.plot_legend('GD','HB','NADAM','FISTA','k-','r-','b-','g-');
end
