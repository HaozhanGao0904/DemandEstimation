function [f, rc] = SN_I_GMM(param,var,option)

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

delta = var.log_s_s0 - log(h_sim./h_sim_0);

W_inv = var.Z_poly'*var.Z_poly;

if option == 1    
    gmmresid = delta - var.Xbeta_hat;        
elseif option == 2
    temp1 = var.X1'*var.Z_poly;
    temp2 = delta'*var.Z_poly;
    beta_tmp = (temp1/W_inv*temp1')\temp1/W_inv*temp2';
    gmmresid = delta - var.X1*beta_tmp;    
end

temp1 = gmmresid'*var.Z_poly;
f = temp1/W_inv*temp1';