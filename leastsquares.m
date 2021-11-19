function [Problem] = leastsquares(A, b, l)
% The definition of the linear least squares problem with L2 regularization.
%
% The problem of interest is defined as
%
%       min f(x) = min||Ax - b||^2 + λ||x||^2 = x^T*A^T*A*x - 2*b^T*A*x + b^T*b + λ*x^T*x.
%       where 
%       x in R^d
%
% Inputs:
%       A           a positive definite matrix of size dxd
%       b           a column vector of size d
%       l           the lambda regularization parameter
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
    Problem.lambda = l
    
    Problem.cost = @cost;
    function f = cost(x)
        f = x'*A'*A*x - 2*b'*A*x + b'*b + l*(x'*x);
    end

    Problem.grad = @grad;
    function d = grad(x)
        d = 2*A'*A*x - 2*A'*b + 2*l*x;
    end

    Problem.grad2 = @grad2;
    function e = grad2()
        e = 2*A'*A;
    end

end