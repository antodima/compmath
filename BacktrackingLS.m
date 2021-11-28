function [as, i] = BacktrackingLS(f, grad_f, x, as, m1, tau, MaxIters)
    i = 1;
    phi0 = f(x);
    g = grad_f(x);
    ng = norm(g);
    phip0 = -ng * ng;
    while i <= MaxIters && as > 1e-16
       x = x - as*g;
       phia = f(x);
       lastg = grad_f(x);
       phip = -g' * lastg;
       if phia <= phi0 + m1*as*phip0  % Armijo satisfied
          break;
       end
       as = as * tau;
       i = i + 1;
    end
end