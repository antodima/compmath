function [x, i] = ACG(Problem, x0, eps, alpha, beta, MaxIter, use_gamma, color, style)

%function [x] = ACG(p, x0, eps, t, MaxIter)
%   Apply the Accelerated Gradient algorithm.

    A = Problem.A;
    b = Problem.b;
    m = Problem.m;
    n = Problem.n;
    f = Problem.cost;
    grad_f = Problem.grad;
    
    x = x0; % starting point
    x1 = x0;
    gamma0 = 0;
    gamma1 = 1;
    
    if Problem.name == "quadratic"
        Problem.plot();
    end
    
    i = 0;
    fprintf( '---Accelerated Gradient method\n');
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
        end
            
        if i == 1
            g = grad_f(x);
            ng = norm(g);
            x = x - alpha*g;
            x1 = x;
        else
            if use_gamma
                gamma1 = (sqrt(4*gamma1^2+gamma1^4)-gamma1^2)/2;
                gamma1 = 1 + sqrt(1+4*gamma0^2);
                beta = gamma1*((1/gamma0)-1);
                beta = (gamma0-1)/gamma1;
            end
            
            y = x + beta*(x1 - x0);
            g = grad_f(y);
            ng = norm(g);
            x = y - alpha*g;
            x0 = x1; x1 = x;
            gamma0 = gamma1;
        end
        
        if Problem.name == "quadratic"
            Problem.plot_line(x0, x1, color, style);
        end
        fprintf('%4d\t v=%1.8e \t ng=%1.4e\n' , i, v, ng);
    end
    
end