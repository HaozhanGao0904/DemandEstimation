function f = cub_spline_1D(x,k)

%x is vector of data: column vector
% k is the number of knots
if k == 0
    f = [ones(size(x)) x x.^2 x.^3];
else
    t = quantile(x,(0:k-1)./k);   
    x_diff = bsxfun(@minus,x,t);
    x_greater = bsxfun(@ge,x,t);
    x_terms = (x_diff.^3).*(x_greater);
    f = [ones(size(x)) x x.^2 x.^3 x_terms];    
end

f = licols(f,1e-10);




