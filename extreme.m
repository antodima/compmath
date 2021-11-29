function [Problem] = extreme(A, b, sigmaType, hiddenDim, lambda)
% MSE with regularization using extreme learning

    d = length(b);
    m = size(A, 1);
    n = size(A, 2);
    h = hiddenDim;
    
    Problem.name = 'extreme';
    Problem.m = m;
    Problem.n = n;
    Problem.samples = d;
    Problem.h = h;
    Problem.Q = A;
    Problem.b = b;
    Problem.l = lambda;
    Problem.W1 = rand(n,h);
    Problem.A = sigma(Problem.Q*Problem.W1, sigmaType);
    Problem.W2 = rand(size(Problem.A,2),1);
    
    Problem.output = @output
    function y = output(x)
        y = Problem.A*x;
    end
    
    Problem.cost = @cost;
    function f = cost(x)
        y = output(x);
        f = mean((Problem.b - y).^2) + Problem.l*(x'*x);
    end

    Problem.grad = @grad;
    function d = grad(x)
        % https://math.stackexchange.com/a/1962938
        y = output(x);
        error = (2 / Problem.n) * (y - Problem.b);
        d = Problem.A'*error - 2*Problem.l*x;
    end
    
    Problem.sigma = @sigma;
    function o = sigma(X, type)
        if type == "sigmoid"
            o = sigmoid(X);
        elseif type == "relu"
            o = relu(X);
        else
            o = linear(X);
        end
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = sigmoid(a)
    y = 1.0 ./ (1 + exp(-a));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = relu(a)
    y = max(0, a);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = linear(x)
   y = x;
end