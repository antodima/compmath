function [L, D, P, x] = LDL(A, b, pivoting)
% LDL factorization
    A_orig = A;
    [m,n] = size(A);
    L = eye(m); D = zeros(m); P = eye(m);
    
    alpha = (1+sqrt(17))/8;
    mu0 = max(max(abs(triu(A))));
    mu1 = max(abs(diag(A)));
    
    if pivoting
        % Bunch–Parlett pivoting
        % https://github.com/hsulab/MatrixFactorization/blob/70da3743eada50506d0a1c3b63274cdb09b1d7f1/LDL.m
        %{%}
        A_diag = diag(A);
        for k=1:n-1
            % find pivot index
            [pivot, p] = max(abs(A_diag(k:n)));
            %p = p + k-1;
            % permutate A
            if p ~= k
                P(:,[k,p]) = P(:,[p,k]);
                A([k,p],:) = A([p,k],:);
                A(:,[k,p]) = A(:,[p,k]);
            end
        end
        

        %{
        if mu1 >= alpha*mu0
            % use 1x1 pivot
            P = eye(n); A_diag = diag(A);
            for k=1:n-1
                % find pivot index
                [pivot, p] = max(abs(A_diag(k:n)));
                % permutate A
                if p ~= k
                    P(:,[k,p]) = P(:,[p,k]);
                    A([k,p],:) = A([p,k],:);
                    A(:,[k,p]) = A(:,[p,k]);
                end
            end
        else
            % use 2x2 pivot
            P = eye(n); P1 = eye(n); P2 = eye(n);
            for k=1:n-1
                % get part of A
                A_sub = A(k:n,k:n);
                % find pivot index
                pivot = 0;
                for i=1:n
                    for j=1:i
                        if abs(A(i,j))>pivot
                            pivot = abs(A(i,j));
                            row_ind = i;
                            col_ind = j;
                        end
                    end
                end
                % permutate A
                if (k~=row_ind) && (k+1~=col_ind)
                    % interchange row/col k and row_ind
                    P1(:,[k,row_ind]) = P1(:,[row_ind,k]);
                    A([k,row_ind],:) = A([row_ind,k],:);
                    A(:,[k,row_ind]) = A(:,[row_ind,k]);
                    % interchange row/col k+1 and col_ind
                    P2(:,[k+1,col_ind]) = P2(:,[col_ind,k+1]);
                    A([k+1,col_ind],:) = A([col_ind,k+1],:);
                    A(:,[k+1,col_ind]) = A(:,[col_ind,k+1]);
                    % calculate transformation matrix P
                    P = P*P1*P2;
                end
            end
        end
        %}
        
    end
    
    % LDL' decomposition
    for k=1:n-1
        D(k, k) = A(k, k);
        L(k+1:end, k) = A(k+1:end, k) / A(k, k);
        A(k+1:end, k+1:end) = A(k+1:end, k+1:end) - L(k+1:end, k) * A(k, k+1:end);
    end
    D(m, n) = A(m, n);
    
    %{
    if pivoting
        disp('-------------')
        fprintf("P\n");
        disp(P);
        fprintf("A\n");
        disp(A_orig);
        fprintf("PLDL'P'\n");
        disp(P*L*D*L'*P');
    end
    %}
    
    L1 = P*L;
    x = L1' \ ((L1\b) ./ diag(D)); % solve the linear system
end