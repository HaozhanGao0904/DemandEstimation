function [f, rc] = SN_I_MD(param,var)

rc = (var.v_SN_I*param)';

exp_x2 = exp(var.p*rc);
exp_u = exp(bsxfun(@plus, var.delta_hat, var.p*rc));
exp_u_cumsum = cumsum(exp_u); 
exp_u_sum1 = exp_u_cumsum(var.cdindex,:);
exp_u_sum1(2:size(exp_u_sum1,1),:) = diff(exp_u_sum1);
denom1 = 1 + exp_u_sum1;
denom = denom1(var.cdid,:);

h_i_tmp = exp_x2./denom;
h_sim = mean(h_i_tmp,2);

h_i_tmp_0 = 1./denom;
h_sim_0 = mean(h_i_tmp_0,2);

g_tmp = var.log_s_s0 - var.delta_hat - log(h_sim./h_sim_0);
f = sum(g_tmp.^2);