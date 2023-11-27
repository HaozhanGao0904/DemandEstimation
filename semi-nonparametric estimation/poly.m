function f = poly(x,k)

f = bsxfun(@power,x,(0:k));
f = licols(f,1e-10);


