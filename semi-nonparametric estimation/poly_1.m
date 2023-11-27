function f = poly_1(x,k)

f = bsxfun(@power,x,(1:k));
f = licols(f,1e-10);


