function [x, i] = FISTA(Problem, x0, eps, MaxIter, color, style)

%function [x] = FISTA(p, x0, eps, t, MaxIter)
%   Apply the FISTA algorithm.

    A = Problem.A;
    b = Problem.b;
    m = Problem.m;
    n = Problem.n;
    f = Problem.cost;
    grad_f = Problem.grad;
    grad2_f = Problem.grad2;
    
    x0 = x0;
    x1 = x0;
    beta0 = 0;
    beta1 = beta0;
    y0 = x0;
    y1 = y0;
    h = grad2_f();  % hessian
    L = max(abs(eig(h)));
    
    if Problem.name == "quadratic"
        Problem.plot();
    end
    
    i = 0;
    fprintf( '---FISTA method\n');
    while true        
        v = f(x0);       % value of the function at x
        g = grad_f(x0);  % gradient at x
        ng = norm(g);   % norm of the gradient
        
        if ng <= eps || i == MaxIter
           break;
        else
            i = i + 1;
        end
        
        g = grad_f(y0);
        ng = norm(g);
        
        beta1 = (1+sqrt(1+4*beta0^2))/2;
        x = y0 - (1/L)*g;
        x1 = x;
        y1 = x1 + ((beta0-1)/beta1)*(x1 - x0);
        
        x0 = x1;
        y0 = y1;
        beta0 = beta1;
        
        if Problem.name == "quadratic"
            Problem.plot_line(x0, x1, color, style);
        end
        fprintf('%4d\t v=%1.8e \t ng=%1.4e\n' , i, v, ng);
    end
    
end