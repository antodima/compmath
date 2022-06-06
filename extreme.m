function [Problem] = extreme(A, b, A_test, b_test, sigmaType, hiddenDim, lambda, use_sign)
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
    Problem.T = A_test;
    Problem.l = lambda;
    Problem.use_sign = use_sign;
    Problem.W1 = rand(n,h);
    Problem.bias = rand(1,1);
    Problem.A = sigma(Problem.Q*Problem.W1, sigmaType);
    Problem.A_test = sigma(Problem.T*Problem.W1, sigmaType);
    Problem.b_test = b_test;
    Problem.W2 = rand(size(Problem.A,2),1);
    
    Problem.output = @output;
    function y = output(x)
        y = Problem.A*x+Problem.bias;
        if Problem.use_sign
            y = sign(y);
        end
    end

    Problem.test = @test;
    function y = test(x)
        y = Problem.A_test*x+Problem.bias;
        if Problem.use_sign
            y = sign(y);
        end
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
        % TODO: gradient with sign
    end

    Problem.grad2 = @grad2;
    function h = grad2()
        % https://math.stackexchange.com/a/1962938
        h = 2*A'*A;
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