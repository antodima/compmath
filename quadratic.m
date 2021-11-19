function [Problem] = quadratic(A, b, interval)
% The definition of the quadratic problem.
%
% The problem of interest is defined as
%
%       min f(x) = 1/2 * x^T * A * x + b^T * x.
%       where 
%       x in R^d
%
% Inputs:
%       A           a positive definite matrix of size dxd
%       b           a column vector of size d
%
% Output:
%       Problem     problem instance. 

    d = length(b);
    m = size(A, 1);
    n = size(A, 2);
    
    if m ~= n
       error('A is not square');
    end
    if m <= 1
       error('A is too small');
    end
    
    Problem.name = 'quadratic';
    Problem.m = m;
    Problem.n = n;
    Problem.dim = d;
    Problem.samples = d;
    Problem.A = A;
    Problem.b = b;
    Problem.interval = interval;
    
    Problem.cost = @cost;
    function f = cost(x)
        f = 0.5 * x'*A*x - b'*x;
    end

    Problem.grad = @grad;
    function d = grad(x)
        d = A*x - b;
    end

    Problem.grad2 = @grad2;
    function e = grad2()
        e = A;
    end
    
    Problem.plot = @plot;
    function [] = plot
        if m == 2
            [XX,YY] = meshgrid(Problem.interval);
            X = XX(:); Y = YY(:); Z = diag(Problem.cost([X Y]'));
            ZZ = reshape(Z, size(XX));
            contour(XX,YY,ZZ);
            hold on;
        end
    end

    Problem.plot_line = @plot_line;
    function [] = plot_line(x1, x2, c)
        if m == 2
            PXY = [x1, x2];
            line('XData', PXY(1 , :), 'YData', PXY(2 , :), 'LineStyle', '-', 'LineWidth', 2, 'Marker', 'o', 'Color', c);
        end
    end

end