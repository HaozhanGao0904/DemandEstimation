function [beta_pre, stderr, res_pre, W_hat] = nonefficient_gmm_linear(y,X,Z)

% Number of observations
n = length(y);

%first step of GMM
W_pre = Z'*Z; %preliminary weighting matrix
beta_pre = ((X'*Z)/W_pre*(Z'*X))\(X'*Z)/W_pre*(Z'*y);

res_pre = y - X*beta_pre;

% Estimated covariance matrix
Q_est = (X'*Z)./n;
g_est = bsxfun(@times,Z,res_pre);
W_hat = cov(g_est).*((n-1)/n);
var_hat = inv(Q_est/W_hat*Q_est');

% Standard errors
stderr  = sqrt(diag(var_hat)./n); 

