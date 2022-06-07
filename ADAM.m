function [x, t] = ADAM(Problem, x0, eps, alpha, beta, MaxIter, use_acc, color, style, verbose)

%function [x] = ADAM(p, x0, eps, t, MaxIter)
%   Apply the Adam algorithm.

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
    
    t = 0;
    beta1 = 0.9; beta2 = 0.999;
    m0 = 0; v0 = 0;
    if verbose == 1
        fprintf( '---Adam method\n');
    end
    while true
        
        v = f(x); % value of the function at x
        if use_acc && t > 0
            y = x + beta*(x1 - x0);
            g = grad_f(y);
            ng = norm(g);
        else
            g = grad_f(x);  % gradient at x
            ng = norm(g);   % norm of the gradient
        end
        
        if ng <= eps || t == MaxIter
           break;
        else
            t = t + 1;
        end
        
        m1 = beta1*m0 + (1 - beta1)*g; % Update biased first moment estimate
        v1 = beta2*v0 + (1 - beta2)*(g.^2); % Update biased second raw moment estimate
        m1_hat = m1 / (1 - beta1*t); % Compute bias-corrected first moment estimate
        v1_hat = v1 / (1 - beta2*t); % Compute bias-corrected second raw moment estimate
        x = x - alpha*m1_hat ./ (v1_hat*(1/2) + 1e-8); % Update parameters
        
        m0 = m1; v0 = v1;
        x0 = x1; x1 = x;
        
        if Problem.name == "quadratic"
            Problem.plot_line(x0, x1, color, style);
        end
        if verbose == 1
            fprintf('%4d\t v=%1.8e \t ng=%1.4e\n' , i, v, ng);
        end
    end
    
end