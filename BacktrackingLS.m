function [as, i] = BacktrackingLS(f, grad_f, x, g, phi0, phip0, as, m1, tau, mina, MaxFeval)
    i = 1;
    while feval <= MaxFeval && as > mina
       x = x - alpha*g;
       phi = f(x);
       lastg = grad_f(x);
       phip = - g' * lastg;
       if phia <= phi0 + m1 * as * phip0  % Armijo satisfied
          break;                          % we are done
       end
       as = as * tau;
       i = i + 1;
    end
end