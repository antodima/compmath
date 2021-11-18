function [x] = GD(Problem, x0, eps, t, MaxIter)

%function [x] = GD(p, x0, eps, t, MaxIter)
%   Apply the Steepest Gradient Descent algorithm.
%

    A = Problem.A;
    b = Problem.b;
    m = Problem.m;
    n = Problem.n;
    f = Problem.cost;
    grad_f = Problem.grad;
    x = x0; % starting point
    disp(Problem.name);
    if Problem.name == "quadratic"
        Problem.plot();
    end
    
    i = 1;
    fprintf( '---Gradient Descent method\n');
    while true        
        v = f(x);       % value of the function at x
        g = grad_f(x);  % gradient at x
        ng = norm(g);   % norm of the gradient
        
        if ng <= eps | i > MaxIter
           break;
        end
        
        if Problem.name == "quadratic"
            den = g'*A*g;
            t = ng^2 / den; % stepsize
            
            x_old = x;
            x = x - t*g;
            Problem.plot_line(x_old, x);
            fprintf('%4d\t v=%1.8e \t ng=%1.4e\n' , i, v, ng);
        else
            x = x - t*g;
            fprintf('%4d\t v=%1.8e \t ng=%1.4e\n' , i, v, ng);
        end
        
        i = i + 1;
    end
    
end