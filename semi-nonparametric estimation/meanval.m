function f = meanval(rc_para,var)

norm = 1;
mvalold = exp(var.delta_0);

rc = rc_para(1) + rc_para(2)*var.v_para;

i = 1;
while (norm > 1e-6 && i<=5000)
	  mval = mvalold.*var.s./mean(ind_sh(log(mvalold),rc,var.p,var),2);
      t = abs(mval-mvalold);
	  norm = max(t);
  	  mvalold = mval;
      i = i + 1;
end

f = log(mval);