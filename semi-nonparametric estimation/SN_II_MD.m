function [f, logit_w]= SN_II_MD(param,var)

poly_w = var.v_SN_II_poly*param;
poly_w = poly_w-(mean(poly_w));
logit_w = exp(poly_w)./sum(exp(poly_w));

exp_x2 = exp(var.p*var.v_SN_II);
exp_u = exp(bsxfun(@plus, var.delta_hat, var.p*var.v_SN_II));
exp_u_cumsum = cumsum(exp_u); 
exp_u_sum1 = exp_u_cumsum(var.cdindex,:);
exp_u_sum1(2:size(exp_u_sum1,1),:) = diff(exp_u_sum1);
denom1 = 1 + exp_u_sum1;
denom = denom1(var.cdid,:);

h_i_tmp = exp_x2./denom;
h_sim = h_i_tmp*logit_w;

h_i_tmp_0 = 1./denom;
h_sim_0 = h_i_tmp_0*logit_w;

g_tmp = var.log_s_s0 - var.delta_hat - log(h_sim./h_sim_0);
f = sum(g_tmp.^2);