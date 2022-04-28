function [x, i] = GD(Problem, x0, eps, t, m1, tau, MaxIter, color, style)

%function [x] = GD(p, x0, eps, t, MaxIter)
%   Apply the Steepest Gradient Descent algorithm.

    A = Problem.A;
    b = Problem.b;
    m = Problem.m;
    n = Problem.n;
    f = Problem.cost;
    grad_f = Problem.grad;
    
    x = x0; % starting point
    
    if Problem.name == "quadratic"
        Problem.plot();
    end
    
    i = 0;
    fprintf( '---Gradient Descent method\n');
    while true        
        v = f(x);       % value of the function at x
        g = grad_f(x);  % gradient at x
        ng = norm(g);   % norm of the gradient
        
        if ng <= eps || i == MaxIter
           break;
        else
            i = i + 1;
        end
        
        if Problem.name == "quadratic"
            den = g'*A*g;
            t = ng^2 / den; % stepsize
            
            x_old = x;
            x = x - t*g;
            Problem.plot_line(x_old, x, color, style);
            fprintf('%4d\t v=%1.8e \t ng=%1.4e\n' , i, v, ng);
        else
            [as, lsiters] = BacktrackingLS(f, grad_f, x, t, m1, tau, 1000);
            
            x = x - as*g;
            fprintf('%4d\t v=%1.8e \t ng=%1.4e \t lsiters=%d \t alpha=%e \n' , i, v, ng, lsiters, as);
        end
    end
    
end