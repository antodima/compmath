function [Problem] = leastsquares(A, b)
% The definition of the least squares problem.
%
% The problem of interest is defined as
%
%       min f(x) = min ||Ax - b||^2 = x^T*A^T*A*x - 2*b^T*A*x + b^T*b.
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
    
    Problem.name = 'leastsquares';
    Problem.m = m;
    Problem.n = n;
    Problem.dim = d;
    Problem.samples = d;
    Problem.A = A;
    Problem.b = b;
    
    Problem.cost = @cost;
    function f = cost(x)
        f = x'*A'*A*x - 2*b'*A*x + b'*b;
    end

    Problem.grad = @grad;
    function d = grad(x)
        d = 2*A'*A*x - 2*A'*b;
    end

    Problem.grad2 = @grad2;
    function e = grad2()
        e = 2*A'*A;
    end

end