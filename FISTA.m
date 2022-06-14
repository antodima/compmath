function [x, i, loss, loss_test, errors, errors_test, rates, norms] = FISTA(Problem, x0, eps, MaxIter, color, style, verbose)

%function [x] = FISTA(p, x0, eps, t, MaxIter)
%   Apply the FISTA algorithm.
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
        Problem.plot_surface();
    end
    
    i = 0;
    if verbose == 1
        fprintf( '---FISTA method\n');
    end
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
        
        norms(end+1) = ng;
        loss(end+1) = v;
        errors(end+1) = norm(b-A*x)/norm(b);
        if isfield(Problem,'A_test')         
            errors_test(end+1) = norm(b_test-A_test*x)/norm(b_test);
            v_test = f_test(x0);
            loss_test(end+1) = v_test;
        end
        
        if Problem.name == "quadratic"
            Problem.plot_line(x0, x1, color, style);
        end
        if verbose == 1
            fprintf('%4d\t v=%1.8e \t ng=%1.4e\n' , i, v, ng);
        end
    end

    fstar = min(loss);
    e = abs(loss - fstar);
    rates = zeros(length(e)-2,1);
    for n = 2:(length(e)-1)
        rates(n-1) = log(e(n+1)/e(n))/log(e(n)/e(n-1));       
    end

%     d = abs(loss - loss(end));
%     for k=1:size(d,2)-2
%         rates(end+1) = log(d(k+1)) / log(d(k));
%     end
    
end