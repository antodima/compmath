function [L, D, P, x] = LDL(A, b, pivoting, strategy)
% LDL factorization
    if nargin < 4
        strategy = 1;
    end
    if pivoting
        fprintf('\nPivot strategy=%d [1=complete, 2=partial]\n', strategy);
    end
  
    [m,n] = size(A);
    if m ~= n
        error('Matrix is not squared!\n');
    end

    L = eye(m); D = zeros(m); P = eye(m);
    
    for k = 1:n-1
        if pivoting
            % https://github.com/hsulab/MatrixFactorization/blob/70da3743eada50506d0a1c3b63274cdb09b1d7f1/LDL.m
            % http://buzzard.ups.edu/courses/2014spring/420projects/math420-UPS-spring-2014-reid-LU-pivoting.pdf
            
            if strategy == 1 % complete pivoting
                B = A(k:n,k:n);
                B_diag = diag(B);
                [pivot, p] = max(B_diag);
                p1 = k-1+p;
                if p1 ~= k
                    P([k,p1],:) = P([p1,k],:);
                    B([k,p],:) = B([p,k],:);
                    B(:,[k,p]) = B(:,[p,k]);
                    A(k:n,k:n) = B;
                    if issymmetric((A+A.')/2)
                        fprintf('A is symmetric! (swap indexes %d and %d)\n', k,p1);
                        disp(A);
                    else
                       fprintf('A is NOT symmetric! (swap indexes %d and %d)\n', k,p1);
                       disp(A);
                    end
                end
                
            elseif strategy == 2 % partial pivoting
                [pivot, p] = max(A(k:n));
                if pivot == 0.0
                    error('The pivot could not be zero.');
                end
                B = A(k:n,k:n);
                p1 = k-1+p;
                if p1 ~= k
                    P([k,p1],:) = P([p1,k],:);
                    B([k,p],:) = B([p,k],:);
                    A(k:n,k:n) = B;
                    if issymmetric((A+A.')/2)
                        fprintf('A is symmetric! (swap indexes %d and %d)\n', k,p1);
                        disp(A);
                    else
                        fprintf('A is NOT symmetric! (swap indexes %d and %d)\n', k,p1);
                        disp(A);
                    end
                end
            end
            
        end
        
        D(k, k) = A(k, k);
        L(k+1:end, k) = A(k+1:end, k) / A(k, k);
        A(k+1:end, k+1:end) = A(k+1:end, k+1:end) - L(k+1:end, k) * A(k, k+1:end);
    end
    D(m, n) = A(m, n);
    L1 = P*L;
    x = L1' \ ((L1\b) ./ diag(D)); % solve the linear system
end