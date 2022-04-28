function [L, D, P, x] = LDL(A, b, pivoting)
% LDL factorization
    m = size(A, 1);
    L = eye(m); D = zeros(m);
    P = eye(m);
    A_diag = diag(A);
    for k = 1:m-1
        % pivoting
        if pivoting
            %[unused, p] = max(abs(A(k:end, k)));
            %p = k-1 + p;
            %P([k,p],:) = P([p,k],:);
            %A=P*A;
            
            % https://github.com/hsulab/MatrixFactorization/blob/70da3743eada50506d0a1c3b63274cdb09b1d7f1/LDL.m
            pivot = max(abs(A_diag(k:m)));
            for j=k:m
                if abs(A_diag(j)) == pivot
                    ind = j;
                end
            end

            if j ~= k
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
    L = P * L;
    x = L' \ ((L\b) ./ diag(D)); % solve the linear system
end