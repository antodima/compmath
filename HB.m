function [x, i] = HB(Problem, x0, eps, alpha, beta, MaxIter, color, style, verbose)

%function [x] = HB(p, x0, eps, alpha, beta, MaxIter)
%   Apply the Heavy Ball algorithm.

    A = Problem.A;
    b = Problem.b;
    m = Problem.m;
    n = Problem.n;
    f = Problem.cost;
    grad_f = Problem.grad;
    
    x = x0; % starting point
    x1 = x0;
    
    if Problem.name == "quadratic"
        Problem.plot_surface();
    end
    
    i = 0;
    if verbose == 1
        fprintf( '---Heavy Ball method\n');
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
        
        if Problem.name == "quadratic"
            den = g'*A*g;
            alpha = ng^2 / den; % stepsize
            
            if i == 1
                x = x - alpha*g;
                x1 = x;
            else
                x = x - alpha*g + beta*(x1 - x0);
                x0 = x1; x1 = x;
            end
            Problem.plot_line(x0, x1, color, style);
            if verbose == 1
                fprintf('%4d\t v=%1.8e \t ng=%1.4e\n' , i, v, ng);
            end
        else
            if i == 1
                x = x - alpha*g;
                x1 = x;
            else
                x = x - alpha*g + beta*(x1 - x0);
                x0 = x1; x1 = x;
            end
            if verbose == 1
                fprintf('%4d\t v=%1.8e \t ng=%1.4e\n' , i, v, ng);
            end
        end
    end
    
end