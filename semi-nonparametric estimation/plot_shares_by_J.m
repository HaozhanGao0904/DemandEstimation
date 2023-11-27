%model:
%random coefficients logit with multiple markets
%two covariates: x and p
%RC's on price
%price is endogenous: perfect competition

clear
close all
close hidden
warning off all

%% read and set parameters

%number of products/markets
JT = 600;

%number of simulations for parametric model
R = 1000;

%fixed parameters in the simulation
%rc parameters
mu = -2;
sigma = .5;
intercept = 0;
beta = 1; %slope, coefficient on x
gamma1 = 1; %coefficient on w in mc
gamma2 = .5; %coefficient on x in mc

%quasi-random seed
q_rc = haltonset(1,'Skip',1e3,'Leap',1e2);
%simulate the rc in the SN estimation
var.v_SN = net(q_rc,R)';
%random draw for rc for parametric estimation
var.v_para = norminv(var.v_SN,0,1);

%true rc: 1-by-R dimension
v = norminv(net(q_rc,10000)',mu,sigma);

%% simulate data
%set seed - changing across simulations
rng(0);

%loop over different J
share_in_true = cell(1,5);
i = 1;
for J = [3 5 10 20] 
    
    %draw variables
    T = JT/J;
    x = randn(JT,1); %exogenous characteristics
    xi = .3*randn(JT,1); %demand shock
    delta = intercept + x*beta + xi; %the linear part of the utility function
    w_c = randn(JT,1); %cost shifter
    xi_c = .1*randn(JT,1); %cost shock
    
    %generate endogenous "price" under perfect competition
    var.p = gamma1*w_c + gamma2*x + xi + xi_c;
    
    %market share
    var.cdid = kron((1:T)',ones(J,1));
    var.cdindex = (J:J:JT)';
    mk_dumm = sparse(dummyvar(var.cdid));
    [sij,si0]= ind_sh(delta,v,var.p,var); %v is the random coefficient on var.p
    
    share_tmp = mean(sij,2);
    share_in_true{1,i} = share_tmp;
    
    i = i + 1;
    
end
t = tiledlayout(2,2);
hist_J1 = nexttile;
histogram(share_in_true{1,1},'Normalization','probability','BinWidth',.01);
title('J=3')

hist_J2 = nexttile;
histogram(share_in_true{1,2},'Normalization','probability','BinWidth',.01);
title('J=5')

hist_J3 = nexttile;
histogram(share_in_true{1,3},'Normalization','probability','BinWidth',.01);
title('J=10')

hist_J4 = nexttile;
histogram(share_in_true{1,4},'Normalization','probability','BinWidth',.01);
title('J=20')

t.Padding = 'compact';
t.TileSpacing = 'compact';
t.XLabel.String = 'Market Share';
t.YLabel.String = 'Probability';
linkaxes([hist_J1 hist_J2 hist_J3 hist_J4],'xy')
