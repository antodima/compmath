function [L, D, x] = LDL(Problem)
% LDL factorization

    A = Problem.A; b = Problem.b;
    m = size(A, 1);
    L = eye(m); D = zeros(m);
    for k = 1:m-1
        D(k, k) = A(k, k);
        L(k+1:end, k) = A(k+1:end, k) / A(k, k);
        A(k+1:end, k+1:end) = A(k+1:end, k+1:end) - L(k+1:end, k) * A(k, k+1:end);
    end
    D(m, m) = A(m, m);
    
    x = L' \ ((L\b) ./ diag(D)); % solve the linear system
end