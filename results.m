clear variables;

%% load problem
load('results/xstar.mat');
load('results/fstar.mat');
load('results/Problem.mat');

%% load results
load('results/grid_losses_fista.mat');
load('results/grid_fista.mat');
load('results/grid_losses_gd.mat');
load('results/grid_gd.mat');

[value, pos] = min(grid_losses_fista);
load(sprintf('results/x%d.mat',pos));
load(sprintf('results/losses%d.mat',pos));
load(sprintf('results/norms%d.mat',pos));
load(sprintf('results/residual%d.mat',pos));
load(sprintf('results/elapsed_time%d.mat',pos));

[value_gd, pos_gd] = min(grid_losses_gd);
load(sprintf('results/losses_gd%d.mat',pos_gd));
load(sprintf('results/norms_gd%d.mat',pos_gd));
load(sprintf('results/residual_gd%d.mat',pos_gd));
load(sprintf('results/elapsed_time_gd%d.mat',pos));

%% plot convergence rate
e1 = abs(losses - fstar);
rates1 = zeros(length(e1)-3,1);
for n=2:(length(e1)-2)
    rates1(n-1) = log(e1(n+1))/log(e1(n));       
end

e2 = abs(losses_gd - fstar);
rates2 = zeros(length(e2)-3,1);
for n=2:(length(e2)-2)
    rates2(n-1) = log(e2(n+1))/log(e2(n));       
end

figure();
plot(rates1,'-','LineWidth',2);
hold on;
plot(rates2,'-','LineWidth',2);
xlabel('t')
ylabel('log|f(x_{t+1}) - f^*| / log|f(x_t) - f^*|');
legend('FISTA','GD');
hold off;
grid on;

%% plot log-loss
figure();
semilogy(losses - fstar, '-', 'LineWidth',2);
hold on;
semilogy(losses_gd - fstar, '-', 'LineWidth',2);
xlabel('t')
ylabel('log( f(x_t) - f^* )');
legend('FISTA','GD');
hold off;
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('results/Problem.mat');
A = Problem.A; b = Problem.b;
I = eye(size(A,2)); AA = A'*A+Problem.l*I; bb = A'*b;

tic; [L1, D1, P1, y1] = LDL(AA, bb, false); t1=toc;
tic; [L2, D2, P2, y2] = LDL(AA, bb, true); t2=toc;
tic; [L3,D3] = ldl(AA); y3 = L3' \ ((L3\bb) ./ diag(D3)); t3=toc;

res_train_1 = norm(bb-AA*y1)/norm(bb);
res_train_2 = norm(bb-AA*y2)/norm(bb);
res_train_3 = norm(bb-AA*y3)/norm(bb);

hold off;

disp("=====================================================================================");
fprintf('Problem parameters: hidden size=%3d, lambda=%1.4e \n', Problem.h, Problem.l);
params = grid_fista(pos,:); h = params(1); MaxIter = params(2); lr = params(3); l = params(4);
fprintf('FISTA best result: \t iterations=%d/%d \t loss=%e \t f*=%e \n', length(losses), MaxIter, value, fstar);
params_gd = grid_gd(pos_gd,:); h = params_gd(1); MaxIter = params_gd(2); lr = params_gd(3); l = params_gd(4);
fprintf('GD best result: \t iterations=%d/%d \t loss=%e \t f*=%e \t lr=%e \n', length(losses_gd), MaxIter, value_gd, fstar, lr);

disp("== CUP results ======================================================================");
fprintf('FISTA \t\t\t | residual=%1.4e | time=%2.5f seconds | f_{t}=%1.4e | f*=%1.4e \n', residual, elapsed_time, losses(end), fstar);
fprintf('GD \t\t\t | residual=%1.4e | time=%2.5f seconds | f_{t}=%1.4e | f*=%1.4e \n', residual_gd, elapsed_time_gd, losses_gd(end), fstar);
fprintf('LDL (matlab) \t\t | residual=%1.4e | time=%2.5f seconds \n', res_train_3, t3);
fprintf('LDL (no pivot) \t\t | residual=%1.4e | time=%2.5f seconds \n', res_train_1, t1);
fprintf('LDL (with pivot) \t | residual=%1.4e | time=%2.5f seconds \n', res_train_2, t2);
disp("=====================================================================================");
