function [x, i, loss, loss_test, errors, errors_test, rates, norms] = GD(Problem, x0, eps, lr, m1, tau, MaxIter, color, style, verbose)

%function [x] = GD(p, x0, eps, t, MaxIter)
%   Apply the Steepest Gradient Descent algorithm.
    loss = []; loss_test = []; 
    errors = []; errors_test = [];
    rates = []; norms = [];

    A = Problem.A;
    b = Problem.b;
    if isfield(Problem,'A_test')
        A_test = Problem.A_test;
        b_test = Problem.b_test;
    end
    m = Problem.m;
    n = Problem.n;
    f = Problem.cost;
    if isfield(Problem,'test')
        t = Problem.test;
        f_test = Problem.cost_test;
    end
    grad_f = Problem.grad;
    
    x = x0; % starting point
    
    if Problem.name == "quadratic"
        Problem.plot_surface();
    end
    
    i = 0;
    if verbose == 1
        fprintf( '---Gradient Descent method\n');
    end
    while true        
        v = f(x);       % value of the function at x
        g = grad_f(x);  % gradient at x
        ng = norm(g);   % norm of the gradient
        
        if ng <= eps || i == MaxIter
           break;
        else
            i = i + 1;
        end

        norms(end+1) = ng;
        loss(end+1) = v;
        errors(end+1) = norm(b-A*x)/norm(b);
        if isfield(Problem,'A_test')         
            errors_test(end+1) = norm(b_test-A_test*x)/norm(b_test);
            v_test = f_test(x0);
            loss_test(end+1) = v_test;
        end
        
        if Problem.name == "quadratic"
            den = g'*A*g;
            lr = ng^2 / den; % stepsize
            
            x_old = x;
            x = x - lr*g;
            Problem.plot_line(x_old, x, color, style);
            if verbose == 1
                fprintf('%4d\t v=%1.8e \t ng=%1.4e\n' , i, v, ng);
            end
        else
            [as, lsiters] = BacktrackingLS(f, grad_f, x, lr, m1, tau, 1000);
            
            x = x - as*g;
            if verbose == 1
                fprintf('%4d\t v=%1.8e \t ng=%1.4e \t lsiters=%d \t alpha=%e \n' , i, v, ng, lsiters, as);
            end
        end
    end
    
end