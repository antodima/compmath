function [x] = HB(Problem, x0, eps, t, beta, MaxIter)

%function [x] = HB(p, x0, eps, t, MaxIter)
%   Apply the Heavy Ball algorithm.
%

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
    
    i = 1;
    fprintf( '---Heavy Ball method\n');
    while true        
        v = f(x);       % value of the function at x
        g = grad_f(x);  % gradient at x
        ng = norm(g);   % norm of the gradient
        
        if ng <= eps | i > MaxIter
           break;
        end
        
        x_old = x;
        if i == 1
            x = x - t*g;
        else
            x = x - t*g + beta*(x - x_old);
        end
        
        if Problem.name == "quadratic"
            Problem.plot_line(x_old, x, 'red');
            fprintf('%4d\t v=%1.8e \t ng=%1.4e\n' , i, v, ng);
        end
        
        i = i + 1;
    end
    
end