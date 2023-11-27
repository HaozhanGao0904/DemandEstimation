function [f, beta] = blp_obj(rc_para,var)

delta = meanval(rc_para,var);

W_inv = var.Z_poly'*var.Z_poly;
temp1 = var.X1'*var.Z_poly;
temp2 = delta'*var.Z_poly;
beta = (temp1*var.invA*temp1')\temp1*var.invA*temp2';
gmmresid = delta - var.X1*beta;
temp3 = gmmresid'*var.Z_poly;
f = temp3/W_inv*temp3';


