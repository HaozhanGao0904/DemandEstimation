function f = legendre_den(x,param)

k = length(param);
n = size(x,1);
%(transformed)Legendre polynomial (Fosgerau 2005)
L = zeros(n,k);
L(:,1) = sqrt(3).*(2*x - 1);
L(:,2) = sqrt(5).*(6*x.^2 - 6*x + 1);
for i = 3:k
    L(:,i) = (sqrt(4*i^2-1)/i)*(2*x - 1).*L(:,i-1) - ((i-1)*sqrt(2*i+1)/(i*sqrt(2*i-3)))*L(:,i-2);
end
poly_terms = 1 + L*param;
deno_const = 1 + sum(param.^2);

f = ((poly_terms).^2)./deno_const;

