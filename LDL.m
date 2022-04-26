function [L, D, P, x] = LDL(A, b, pivoting)
% LDL factorization
    m = size(A, 1);
    L = eye(m); D = zeros(m);
    P = 1:m;
    for k = 1:m-1
        % pivoting
        if pivoting
            [unused, p] = max(abs(A(k:end, k)));
            p = k-1 + p;
            A([k,p], 1:end) = A([p,k], 1:end);
            P([k, p]) = P([p, k]);
        end
        
        D(k, k) = A(k, k);
        L(k+1:end, k) = A(k+1:end, k) / A(k, k);
        A(k+1:end, k+1:end) = A(k+1:end, k+1:end) - L(k+1:end, k) * A(k, k+1:end);
    end
    D(m, m) = A(m, m);
    if pivoting
        L = L(P,:);
    end
    x = L' \ ((L\b) ./ diag(D)); % solve the linear system
end