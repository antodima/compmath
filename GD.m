function [x, i, losses, norms] = GD(Problem, x0, fstar, eps, lr, m1, tau, MaxIter, color, style, verbose)

%function [x] = GD(p, x0, eps, t, MaxIter)
%   Apply the Steepest Gradient Descent algorithm.
    losses = []; norms = [];

    A = Problem.A;
    b = Problem.b;
    m = Problem.m;
    n = Problem.n;
    f = Problem.cost;
    grad_f = Problem.grad;
    grad2_f = Problem.grad2;
    
    x = x0; % starting point
    h = grad2_f();  % hessian
    L = max(abs(eig(h)));
    
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
        
        % relative_error = (v - fstar)/abs(fstar);
        absolute_error = (v - fstar);
        if absolute_error <= eps || i == MaxIter
           break;
        else
            i = i + 1;
        end

        losses(end+1) = v;
        norms(end+1) = ng;
        
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
            % [as, lsiters] = BacktrackingLS(f, grad_f, x, lr, m1, tau, 1000);
            % x = x - as*g;
            %x = x - lr*g;
            x = x - (1/L)*g;

            if verbose == 1
                fprintf('%4d\t v=%1.8e \t ng=%1.4e \t lr=%e \n' , i, v, ng, lr);
            end
        end
    end
    
end