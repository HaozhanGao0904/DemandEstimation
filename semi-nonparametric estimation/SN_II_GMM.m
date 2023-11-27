function [f, logit_w]= SN_II_GMM(param,var,option)

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