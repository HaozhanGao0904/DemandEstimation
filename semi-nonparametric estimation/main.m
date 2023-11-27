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
%read parameters
params = dlmread('parameter.txt');

%number of markets
T = params(2);

%number of products
J = params(3);

%order of sieve for approximating \phi(p)
K1 = params(4);

%order of sieve for constructing IV
K2 = params(5);

%order of sieve for approximating F
K3_I = params(6);
K3_II = params(7);
K3_III = params(8);

%number of simulations for parametric model
R = params(9);

%fixed parameters in the simulation
%rc parameters
mu = -2;
sigma = .5;
intercept = -10;
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
rng(params(1));

%draw variables
JT = J*T;
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
var.s = mean(sij,2);
var.s0 = mean(si0,2);

%% prepare for estimation
p_poly_tmp = poly_1(var.p,K1);
w_poly_tmp = cub_spline_1D(w_c,K2);
p_poly = [];
w_poly = [];
for t = 1:T
    p_poly = [p_poly bsxfun(@times,p_poly_tmp,mk_dumm(:,t))];  %#ok<AGROW>
    w_poly = [w_poly bsxfun(@times,w_poly_tmp,mk_dumm(:,t))];  %#ok<AGROW>
end
p_poly = licols(p_poly,1e-10);
X_poly = [x ones(JT,1) p_poly];

var.X1 = [x ones(JT,1)];
Z_tmp = [cub_spline_1D(x,K2) w_poly];
var.Z_poly = licols(Z_tmp,1e-10);

%optimization options
options_fminbnd = optimset(...
    'MaxFunEvals', 10000,...
    'TolFun',      1e-6,...
    'TolX',        1e-6,...
    'MaxIter',     500,...
    'Display','iter');
options = optimoptions('fmincon','Display','iter', 'MaxFunEvals',100000,'MaxIter',500);

%% BLP estimator
var.invA = inv(var.Z_poly'*var.Z_poly);
var.delta_0 = delta; %starting value for solving contraction mapping
rc_para_0 = [mu;sigma];
LB = [mu-5;-sigma-3];
UB = [mu+5; sigma+3];
[rc_est_BLP,~,~,~] = fminsearchbnd(@(y)blp_obj(y,var),rc_para_0,LB,UB,options_fminbnd);
[~, beta_BLP] = blp_obj(rc_est_BLP,var);

%% SN estimator - first stage
var.log_s_s0 = log(var.s./var.s0);
[coef_SN, se_non, xi_hat, var.W_hat] = efficient_gmm_linear(var.log_s_s0,X_poly,var.Z_poly);
beta_SN = coef_SN(1:2);
se_SN = se_non(1:2);
var.delta_hat = X_poly(:,1:2)*beta_SN +  xi_hat; %fitted delta
var.Xbeta_hat = X_poly(:,1:2)*beta_SN; %fitted X*beta

%% SN estimator - parametric estimation (imposing normality)
rc_est_SNpara_MD = fmincon(@(y)SNpara_MD(y,var),rc_para_0,[],[],[],[],LB,UB,[],options);
rc_est_SNpara_GMM1 = fmincon(@(y)SNpara_GMM(y,var,1),rc_para_0,[],[],[],[],LB,UB,[],options);
rc_est_SNpara_GMM2 = fmincon(@(y)SNpara_GMM(y,var,2),rc_para_0,[],[],[],[],LB,UB,[],options);

%% SN estimator - sieve I£º approximate inverse CDF
var.v_SN_I = poly(var.v_SN',K3_I);
rc_SN_I_0 = [-1; 0*ones(K3_I,1)]; %starting values
LB_I = [-10;-inf*ones(K3_I,1)];
UB_I = [10;  inf*ones(K3_I,1)];

rc_SN_I_MD = fmincon(@(y)SN_I_MD(y,var),rc_SN_I_0,[],[],[],[],LB_I,UB_I,[],options);
[~,v_SN_I_MD] = SN_I_MD(rc_SN_I_MD,var);
rc_mean_SN_I_MD = nanmean(v_SN_I_MD);
rc_std_SN_I_MD = nanstd(v_SN_I_MD);

rc_SN_I_GMM1 = fmincon(@(y)SN_I_GMM(y,var,1),rc_SN_I_0,[],[],[],[],LB_I,UB_I,[],options);
[~,v_SN_I_GMM1] = SN_I_GMM(rc_SN_I_GMM1,var,1);
rc_mean_SN_I_GMM1 = nanmean(v_SN_I_GMM1);
rc_std_SN_I_GMM1 = nanstd(v_SN_I_GMM1);

rc_SN_I_GMM2 = fmincon(@(y)SN_I_GMM(y,var,2),rc_SN_I_0,[],[],[],[],LB_I,UB_I,[],options);
[~,v_SN_I_GMM2] = SN_I_GMM(rc_SN_I_GMM2,var,2);
rc_mean_SN_I_GMM2 = nanmean(v_SN_I_GMM2);
rc_std_SN_I_GMM2 = nanstd(v_SN_I_GMM2);

%% SN estimator - sieve II: Train's sieve
range_v = 10*sigma;
var.v_SN_II = (mu-range_v/2) + range_v*var.v_SN(1:100);
var.v_SN_II_poly = poly(var.v_SN_II',K3_II);
rc_SN_II_0 = 0*ones(size(var.v_SN_II_poly,2),1);

rc_SN_II_MD = fmincon(@(y)SN_II_MD(y,var),rc_SN_II_0,[],[],[],[],[],[],[],options);          
[~,w_SN_II_MD] = SN_II_MD(rc_SN_II_MD,var);
rc_mean_SN_II_MD = var.v_SN_II*w_SN_II_MD;
rc_std_SN_II_MD = sqrt(((var.v_SN_II - rc_mean_SN_II_MD).^2)*w_SN_II_MD);

rc_SN_II_GMM1 = fmincon(@(y)SN_II_GMM(y,var,1),rc_SN_II_0,[],[],[],[],[],[],[],options);          
[~,w_SN_II_GMM1] = SN_II_GMM(rc_SN_II_GMM1,var,1);
rc_mean_SN_II_GMM1 = var.v_SN_II*w_SN_II_GMM1;
rc_std_SN_II_GMM1 = sqrt(((var.v_SN_II - rc_mean_SN_II_GMM1).^2)*w_SN_II_GMM1);

rc_SN_II_GMM2 = fmincon(@(y)SN_II_GMM(y,var,2),rc_SN_II_0,[],[],[],[],[],[],[],options);          
[~,w_SN_II_GMM2] = SN_II_GMM(rc_SN_II_GMM2,var,2);
rc_mean_SN_II_GMM2 = var.v_SN_II*w_SN_II_GMM2;
rc_std_SN_II_GMM2 = sqrt(((var.v_SN_II - rc_mean_SN_II_GMM2).^2)*w_SN_II_GMM2);

%% SN estimator - sieve III: Fosgerau and Bierlaire (2007)          

rc_SN_III_0 = [mu;sigma;0*ones(K3_III,1)];
LB_III = [-10; 0; -10*ones(K3_III,1)];
UB_III = [10; 10; 10*ones(K3_III,1)];

rc_SN_III_MD = fmincon(@(y)SN_III_MD(y,var),rc_SN_III_0,[],[],[],[],LB_III,UB_III,[],options);          
[~,v_SN_III_MD,w_SN_III_MD] = SN_III_MD(rc_SN_III_MD,var);
rc_mean_SN_III_MD = (v_SN_III_MD*w_SN_III_MD)./R;
rc_std_SN_III_MD = sqrt((((v_SN_III_MD - rc_mean_SN_III_MD).^2)*w_SN_III_MD)./R);

rc_SN_III_GMM1 = fmincon(@(y)SN_III_GMM(y,var,1),rc_SN_III_0,[],[],[],[],[],[],[],options);          
[~,v_SN_III_GMM1,w_SN_III_GMM1] = SN_III_GMM(rc_SN_III_GMM1,var,1);
rc_mean_SN_III_GMM1 = (v_SN_III_GMM1*w_SN_III_GMM1)./R;
rc_std_SN_III_GMM1 = sqrt((((v_SN_III_GMM1 - rc_mean_SN_III_GMM1).^2)*w_SN_III_GMM1)./R);

rc_SN_III_GMM2 = fmincon(@(y)SN_III_GMM(y,var,2),rc_SN_III_0,[],[],[],[],[],[],[],options);          
[~,v_SN_III_GMM2,w_SN_III_GMM2] = SN_III_GMM(rc_SN_III_GMM2,var,2);
rc_mean_SN_III_GMM2 = (v_SN_III_GMM2*w_SN_III_GMM2)./R;
rc_std_SN_III_GMM2 = sqrt((((v_SN_III_GMM2 - rc_mean_SN_III_GMM2).^2)*w_SN_III_GMM2)./R);

results = ['mc_SN_RC_Logit_', num2str(params(1))];
save(results, 'params',...
    'rc_est_BLP','beta_BLP',...
    'beta_SN','se_SN',...
    'rc_est_SNpara_MD','rc_est_SNpara_GMM1','rc_est_SNpara_GMM2',...
    'rc_mean_SN_I_MD','rc_std_SN_I_MD','rc_mean_SN_I_GMM1','rc_std_SN_I_GMM1','rc_mean_SN_I_GMM2','rc_std_SN_I_GMM2',...
    'rc_mean_SN_II_MD','rc_std_SN_II_MD','rc_mean_SN_II_GMM1','rc_std_SN_II_GMM1','rc_mean_SN_II_GMM2','rc_std_SN_II_GMM2',...
    'rc_mean_SN_III_MD','rc_std_SN_III_MD','rc_mean_SN_III_GMM1','rc_std_SN_III_GMM1','rc_mean_SN_III_GMM2','rc_std_SN_III_GMM2'...
    );