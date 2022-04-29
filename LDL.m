function [L, D, P, x] = LDL(A, b, pivoting)
% LDL factorization
    m = size(A, 1);
    L = eye(m); D = zeros(m);
    P = eye(m); A_diag = diag(A);
    for k = 1:m-1
        if pivoting
            % https://github.com/hsulab/MatrixFactorization/blob/70da3743eada50506d0a1c3b63274cdb09b1d7f1/LDL.m
            %A_diag = diag(A);
            [pivot, ind] = max(abs(A_diag(k:m)));
            ind = k-1 + ind;
            if ind ~= k
                P(:,[k,ind]) = P(:,[ind,k]);
                A([k,ind],:) = A([ind,k],:);
                A(:,[k,ind]) = A(:,[ind,k]);
            end
        end
        
        D(k, k) = A(k, k);
        L(k+1:end, k) = A(k+1:end, k) / A(k, k);
        A(k+1:end, k+1:end) = A(k+1:end, k+1:end) - L(k+1:end, k) * A(k, k+1:end);
    end
    D(m, m) = A(m, m);
    L = P*L;
    x = L' \ ((L\b) ./ diag(D)); % solve the linear system
end