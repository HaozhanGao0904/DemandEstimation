function [s,s0]= ind_sh(delta,v,x,var)

exp_u = exp(bsxfun(@plus, delta, x*v));
exp_u_cumsum = cumsum(exp_u); 
exp_u_sum1 = exp_u_cumsum(var.cdindex,:);
exp_u_sum1(2:size(exp_u_sum1,1),:) = diff(exp_u_sum1);
denom1 = 1 + exp_u_sum1;
denom = denom1(var.cdid,:);
s = exp_u./denom;
s0 = 1./denom;