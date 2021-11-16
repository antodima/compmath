function [x] = SGD(A, b, x0, eps, MaxIter)

%function [ x ] = SGD(A, b, x0, eps, MaxIter)
%   Apply the Steepest Gradient Descent algorithm with exact line search to the
%   quadratic function
%
%   f(x) = 1/2 x^T Q x + q x
%
%   Input:
%
%       - A ([ n x n ] real symmetric matrix, not necessarily positive
%           semidefinite): the Hessian (quadratic part) of f
%
%       - b ([ n x 1 ] real column vector): the linear part of f
%
%       - x0 ([ n x 1 ] real column vector): the point where to start the
%           algorithm from.
%
%       - eps (real scalar, optional, default value 1e-6): the accuracy in 
%           the stopping criterion: the algorithm is stopped when the norm 
%           of the gradient is less than or equal to eps
%
%       - MaxIter (integer scalar, optional, default value 1000): the
%           maximum number of iterations

    i = 1;
    x = x0; % starting point
    fprintf( 'Steepest Gradient method\n');
    while true
        v = 0.5 * x'*A*x + b'*x; % the value of the quadratic function at x
        g = A*x + b;             % gradent of the quadratic function
        ng = norm(g);            % norm of the gradient
        
        den = g'*A*g;
        t = ng^2 / den; % stapsize        
        x = x - t*g;
        fprintf('%4d\t v: %1.8e \t ng: %1.4e\n' , i, v, ng);
        
        if ng <= eps
           break;
        end
        i = i + 1;
    end
end