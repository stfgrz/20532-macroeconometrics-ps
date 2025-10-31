%% 20532 Macroeconometrics | Problem Set 2
%
% ---------------------------------------------------------------
% Author: Stefano Graziosi
% Date: 2025-09-22
% ---------------------------------------------------------------

%% Housekeeping & graphics style 
clear; clc; close all; format compact
outdir = fullfile(pwd,'ps2/output');                                        % Output folder
if ~exist(outdir,'dir'), mkdir(outdir); end                                 % Well if it doesn't exist, create it

% Clean, consistent figure defaults
set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLineLineWidth', 1.5);

% Helper to export figures
exportFig = @(fh, name) exportgraphics(fh, fullfile(outdir, name), 'ContentType','vector');

%% Helper functions

% Q1
simulate_rw        = @(T, sig2) cumsum([0; sqrt(sig2)*randn(T-1,1)]);       % y_t = y_{t-1}+eps_t, y_1=0
ols     = @(y,x) deal((x'*x)\(x'*y), y - x*((x'*x)\(x'*y)));                % generic OLS

% (Local functions moved to the end of this script to satisfy MATLAB's
% requirement that local functions appear after all script code.)

% Global settings
z_5_left = -1.6448536269;                               % 5% one-sided Normal critical (left tail)
% chi2_95 is defined later where it is used to avoid unused-variable warnings
DF_tau_mu = struct('p1',-3.43,'p5',-2.86,'p10',-2.57);  % common tau_mu tabulated values

%% ===================== Exercise 1 =====================

%%% Question (a):
%   Compute the empirical distribution of the OLS estimator in the case of an AR(1) with $\varphi = 1$ and $T=250$ (you are free to choose the variance of the innovation).

rng(20532,'twister');        % reproducibility
T        = 250;              % sample length
R        = 5000;             % Monte Carlo replications
sigma2   = 0.6;              % variance of innovations (free choice)

phi_hat_a = zeros(R,1);

for r=1:R
    y = simulate_rw(T, sigma2);
    yt = y(2:end);  
    x = y(1:end-1);
    bhat = (x'*x)\(x'*yt);
    phi_hat_a(r) = bhat;
end

% Plot histogram (standardized size and bins)
figPos = [100 100 840 420];
histBins = 50;
fh_a = figure('Position', figPos);
histogram(phi_hat_a, histBins, 'Normalization','pdf'); grid on; hold on
xline(1,'--','True $\phi=1$','LabelVerticalAlignment','bottom');
xlabel('$\hat{\phi}$'); ylabel('Density')
title('(a) Empirical distribution of OLS $\hat{\phi}$ under unit root ($T=250$)')
drawnow;
exportFig(fh_a,'1a_phi_hat_hist.pdf');
close(fh_a);

fprintf('(a) mean(\\hat{phi})=%.4f, sd=%.4f, med=%.4f\n', mean(phi_hat_a), std(phi_hat_a), median(phi_hat_a));

%%% Question (b):
%   Repeat the exercise in (a) but now with a drift term equal to Construct a $t$-test for the null hypothesis $H_{0}:\ \rho=\varphi-1=0$, in a test regression: $\Delta y_{t}=\alpha+\rho y_{t-1}+\varepsilon_{t}$; against a one-sided alternative $H_{0}:\ \rho<0$.
%   Using a standard Normal distribution, how often do you reject the null hypothesis at the $95\%$ confidence level?
%   Is the actual distribution of the t-test symmetric? Discuss.

t_b   = zeros(R,1);
rho_b = zeros(R,1);

for r=1:R
    y    = simulate_rw(T, sigma2);
    dy   = diff(y);
    ylag = y(1:end-1);
    X    = [ones(T-1,1), ylag];
    [b, u] = ols(dy, X);
    nu   = (T-1) - size(X,2);                   % df = T-1 - 2
    s2   = (u'*u)/nu;
    Vb   = s2 * inv(X'*X);
    rho_hat = b(2);
    se_rho  = sqrt(Vb(2,2));
    t_b(r)  = rho_hat / se_rho;
    rho_b(r)= rho_hat;
end

rej_norm_left = mean(t_b < z_5_left);           % one-sided at 5% using N(0,1)
sk_b = skewness(t_b);

fprintf(['(b) Using Normal 5%% one-sided (z=%.3f): reject rate = %.3f. ', ...
         'Skewness of t-stat = %.3f (non-symmetric).\n'], z_5_left, rej_norm_left, sk_b);

% Histogram of t_b for illustration
% Histogram of t-stat (standardized)
fh_b = figure('Position', figPos);
histogram(t_b, histBins, 'Normalization','pdf'); grid on; hold on
xline(z_5_left, '--', 'Normal 5% one-sided crit','LabelVerticalAlignment','bottom');
xlabel('$t(\hat{\rho})$'); ylabel('Density');
title('(b) DF $t$-stat under unit root, intercept included')
drawnow;
exportFig(fh_b,'1b_t_hist.pdf');
close(fh_b);

%%% Question (c):
%   Compute now few percentiles of the empirical distribution of the $t$-test you generated at point b. and check that they are close to those tabulated by Dickey and Fuller.

% Dickey–Fuller "tabulated" criticals for T~250
% tau_mu (intercept only, no trend) and tau_trend (intercept + trend)
DFcrit.tau_mu   = struct('p1',-3.46,'p5',-2.88,'p10',-2.57);
DFcrit.tau_trend= struct('p1',-3.99,'p5',-3.43,'p10',-3.12);

pct_vec = [1 5 10 25 50 75 90 95 99];
emp_pct = prctile(t_b, pct_vec);
DF_tab  = [DFcrit.tau_mu.p1, DFcrit.tau_mu.p5, DFcrit.tau_mu.p10]; % 1%,5%,10% (left tail)

% Print a compact comparison
fprintf('(c) Empirical DF t percentiles (%%):\n');
disp(table(pct_vec(:), emp_pct(:), 'VariableNames',{'percentile','empirical_t'}));
fprintf('Tabulated DF (tau_mu) ~ T=250: 1%%=%.2f, 5%%=%.2f, 10%%=%.2f\n', ...
        DFcrit.tau_mu.p1, DFcrit.tau_mu.p5, DFcrit.tau_mu.p10);

% Save CSV
writetable(table(pct_vec(:), emp_pct(:), 'VariableNames',{'percentile','empirical_t'}), ...
           fullfile(outdir,'1c_empirical_t_percentiles.csv'));

% Compare left-tail 1/5/10% to DF τ_μ and report the gaps (empirical − table)
left_tail   = [1 5 10];
emp_left    = prctile(t_b, left_tail);
DF_left     = [DFcrit.tau_mu.p1; DFcrit.tau_mu.p5; DFcrit.tau_mu.p10];
delta_left  = emp_left(:) - DF_left;

comp_tbl = table(left_tail(:), emp_left(:), DF_left, delta_left, ...
    'VariableNames', {'percentile','empirical_t','DF_tau_mu','emp_minus_tab'});

disp('Comparison to DF \tau_\mu at 1/5/10% (empirical − table):');
disp(comp_tbl);

% Check rejection using DF 5% critical (should be about 0.05 under H0)
rej_DF5 = mean(t_b < DFcrit.tau_mu.p5);
fprintf('Using DF \\tau_\\mu 5%% critical (%.2f): empirical rejection = %.3f\n', ...
        DFcrit.tau_mu.p5, rej_DF5);

% Save comparison table
writetable(comp_tbl, fullfile(outdir,'1c_empirical_vs_DF_tau_mu.csv'));

%%% Question (d)
% Compute the empirical distribution of the OLS in the case of a random walk with drift and $T = 250$ and study the performance of the Dickey--Fuller test.

rng(20532,'twister');                      % reproducibility
a0_drift = 0.5;                            % drift (Enders' example uses 0.5)
R        = 5000;                           % Monte Carlo reps (reuse if defined above)
T        = 250;                            % sample length (reuse if defined above)
sigma2   = 0.6;                            % variance (reuse if defined above)

simulate_rw_drift = @(T, sig2, mu) cumsum([0; mu + sqrt(sig2)*randn(T-1,1)]);

phi_hat_d  = zeros(R,1);                   % OLS slope from y_t on [1, y_{t-1}]
alpha_hat_d= zeros(R,1);                   % OLS intercept
t_d        = zeros(R,1);                   % DF t-stat on ρ
rho_d      = zeros(R,1);                   % ρ-hat

for r=1:R
    y    = simulate_rw_drift(T, sigma2, a0_drift);
    % OLS of y_t on [1, y_{t-1}] to look at empirical dist of \hat{\phi} under drifted RW
    yt   = y(2:end);
    xlag = y(1:end-1);
    X01  = [ones(T-1,1), xlag];
    [b01,u01] = ols(yt, X01);                 
    alpha_hat_d(r) = b01(1);
    phi_hat_d(r)   = b01(2);

    % Dickey–Fuller regression with intercept (\tau_{\mu} case)
    dy   = diff(y);
    Xdf  = [ones(T-1,1), xlag];
    [b,u] = ols(dy, Xdf);
    nu   = (T-1) - size(Xdf,2);               % df = T-1 - 2
    s2   = (u'*u)/nu;
    Vb   = s2 * inv(Xdf'*Xdf);
    rho_hat = b(2);
    se_rho  = sqrt(Vb(2,2));
    t_d(r)  = rho_hat / se_rho;
    rho_d(r)= rho_hat;
end

% Plot empirical dist of \hat\phi
figPos   = [100 100 840 420];
histBins = 50;
fh_d1 = figure('Position', figPos);
histogram(phi_hat_d, histBins, 'Normalization','pdf'); grid on; hold on
xline(1,'--','True $\phi=1$','LabelVerticalAlignment','bottom');
xlabel('$\hat{\phi}$'); ylabel('Density');
title('(d) OLS $\hat{\phi}$ under RW+drift ($T=250$)')
exportFig(fh_d1,'1d_phi_hat_hist.pdf'); close(fh_d1);

% DF rejection using Normal vs DF (τ_μ) 5% left-tail
z_5_left   = -1.6448536269;                             % from your header
rej_norm_d = mean(t_d < z_5_left);
rej_DF5_d  = mean(t_d < DFcrit.tau_mu.p5);              % τ_μ criticals from your (c)

fprintf(['(d) OLS under RW+drift: mean(phi-hat)=%.4f, sd=%.4f.\n' ...
         '    DF (one-sided) reject@5%% using Normal: %.3f; using DF τ_μ: %.3f\n'], ...
        mean(phi_hat_d), std(phi_hat_d), rej_norm_d, rej_DF5_d);

writetable(table(phi_hat_d, alpha_hat_d, t_d, rho_d), fullfile(outdir,'1d_rw_drift_results.csv'));

%%% Question(e)
%Construct an F-test for the null hypothesis $H_{0}$: there is unit root, against the alternative $H_{1}$: there is no unit root using a $\chi^{2}$ distribution (how many degrees of freedom?). How often do you reject $H_{0}$ at 95\% confidence?

% Wald test on ρ=0 using χ^2(1) ~ t^2 (large-sample). Report size under H0.
chi2_95 = 3.8414588207;                        % from your header
chi2_stat = t_d.^2;                             % t^2 ~ \Chi^2(1)
rej_chi2  = mean(chi2_stat > chi2_95);

fprintf('(e) Wald χ^2 test (df=1) reject@95%% under H0 (RW+drift): %.3f\n', rej_chi2);

% (Optional) visualize t and χ^2 stats
fh_e1 = figure('Position', figPos);
histogram(t_d, histBins, 'Normalization','pdf'); grid on; hold on
xline(-sqrt(chi2_95),'--','$\pm \sqrt{\chi^2_{0.95;1}}$','LabelVerticalAlignment','bottom');
xline(+sqrt(chi2_95),'--');
xlabel('$t(\hat{\rho})$'); ylabel('Density'); title('(e) DF t-stats under H_0 (RW+drift)')
exportFig(fh_e1,'1e_t_hist_RWdrift.pdf'); close(fh_e1);

%%% Question (f)
% Generate now data from a deterministic time trend and perform a DF test using the correct distribution for the test with null hypothesis $H_{0}$: there is unit root. How often do you reject the null? \emph{(hint: you can find additional details in Enders).}

beta0 = 0.0; 
beta1 = 0.05;                                   % slope of deterministic trend
R      = 5000;
tvec   = (1:T)';

t_f   = zeros(R,1);                             % t-stat on \rho with trend
rho_f = zeros(R,1);

for r=1:R
    eps = sqrt(sigma2)*randn(T,1);
    y   = beta0 + beta1*tvec + eps;             % trend-stationary, iid errors
    dy  = diff(y);
    ylag= y(1:end-1);
    t2  = tvec(2:end);
    Xtr = [ones(T-1,1), t2, ylag];
    [b,u] = ols(dy, Xtr);
    nu   = (T-1) - size(Xtr,2);
    s2   = (u'*u)/nu;
    Vb   = s2 * inv(Xtr'*Xtr);
    rho_hat = b(3);
    se_rho  = sqrt(Vb(3,3));
    t_f(r)  = rho_hat / se_rho;
    rho_f(r)= rho_hat;
end

% Use τ_τ 5% left-tail (with trend)
rej_trend_DF5 = mean(t_f < DFcrit.tau_trend.p5);

fprintf(['(f) Trend-stationary DGP: DF with trend (τ_τ) reject@5%% = %.3f ', ...
         '(power against unit root).\n'], rej_trend_DF5);

% Save summary
writetable(table(t_f, rho_f), fullfile(outdir,'1f_trendstationary_results.csv'));
%% ===================== Exercise 2 =====================

%%% Monte Carlo settings
T   = 250;         % sample size
R   = 2000;        % replications (raise if you like)
B   = 500;         % burn-in where needed
nu  = T - 2;       % df for slope t-test with constant

if exist('tinv','file')
    tcrit = tinv(0.975, nu);     % two-sided 5%
else
    tcrit = 1.96;                % Normal approx
end

%% Containers (one row per Case)
caseNames = ["Case 1: both stationary", ...
             "Case 2: I(1) vs I(0)", ...
             "Case 3: I(1) vs I(1) (spurious)", ...
             "Case 4: I(1) & cointegrated"];

K = numel(caseNames);
mean_b1   = zeros(K,1);
sd_b1     = zeros(K,1);
rej_rate  = zeros(K,1);
mean_R2   = zeros(K,1);
mean_rhoU = zeros(K,1);      % residual lag-1 corr
mean_DW   = zeros(K,1);      % Durbin–Watson

% Store full draws for distributions (for plots)
t_all  = cell(K,1);
R2_all = cell(K,1);

%% ---------- CASE 1: both stationary (OLS is meaningful) ----------
% y_t = 0.5 y_{t-1} + e^y_t,  z_t = -0.3 z_{t-1} + e^z_t  (independent)
phi_y = 0.5; phi_z = -0.3;
sig2y = 1.0; sig2z = 1.0;

[b1, tstat, R2, rhoU, DW] = deal(zeros(R,1));
for r = 1:R
    yy = simulate_ar1_loop(T+B, phi_y, sig2y, 0, 0);
    zz = simulate_ar1_loop(T+B, phi_z, sig2z, 0, 0);
    y = yy(B+1:end); z = zz(B+1:end);

    [b, se, tvec, R2r, u, DWr] = ols_with_const(y, z);
    b1(r)   = b(2);
    tstat(r)= tvec(2);
    R2(r)   = R2r;
    rhoU(r) = lag1corr(u);
    DW(r)   = DWr;
end
[mean_b1(1), sd_b1(1), rej_rate(1), mean_R2(1), mean_rhoU(1), mean_DW(1)] = ...
    summarize_case(b1, tstat, R2, rhoU, DW, tcrit);
t_all{1}  = tstat; R2_all{1} = R2;

%% ---------- CASE 2: I(1) vs I(0) (meaningless) ----------
% y_t is random walk; z_t stationary AR(1)
phi_z = 0.6;
for r = 1:R
    y = cumsum(randn(T,1));                 % I(1), variance 1
    zz = simulate_ar1_loop(T+B, phi_z, 1.0, 0, 0);
    z = zz(B+1:end);

    [b, se, tvec, R2r, u, DWr] = ols_with_const(y, z);
    b1(r)   = b(2);
    tstat(r)= tvec(2);
    R2(r)   = R2r;
    rhoU(r) = lag1corr(u);
    DW(r)   = DWr;
end
[mean_b1(2), sd_b1(2), rej_rate(2), mean_R2(2), mean_rhoU(2), mean_DW(2)] = ...
    summarize_case(b1, tstat, R2, rhoU, DW, tcrit);
t_all{2}  = tstat; R2_all{2} = R2;

%% ---------- CASE 3: I(1) vs I(1) independent (classic spurious) ----------
for r = 1:R
    y = cumsum(randn(T,1));     % I(1)
    z = cumsum(randn(T,1));     % I(1), independent

    [b, se, tvec, R2r, u, DWr] = ols_with_const(y, z);
    b1(r)   = b(2);
    tstat(r)= tvec(2);
    R2(r)   = R2r;
    rhoU(r) = lag1corr(u);
    DW(r)   = DWr;
end
[mean_b1(3), sd_b1(3), rej_rate(3), mean_R2(3), mean_rhoU(3), mean_DW(3)] = ...
    summarize_case(b1, tstat, R2, rhoU, DW, tcrit);
t_all{3}  = tstat; R2_all{3} = R2;

% Double check: differencing fixes the spurious regression in Case 3
rej_diff = nan; %#ok<NASGU>
do_diff_fix = true;
if do_diff_fix
    tstatD = zeros(R,1);
    for r = 1:R
        y = cumsum(randn(T,1));
        z = cumsum(randn(T,1));
        dy = diff(y); dz = diff(z);
        [~, ~, tvec, ~] = ols_with_const(dy, dz);
        tstatD(r) = tvec(2);
    end
    rej_diff = mean(abs(tstatD) > tcrit);
end

%% ---------- CASE 4: I(1) & cointegrated (meaningful despite I(1)) ----------
% y_t = tau_t + eps_y,t,  z_t = tau_t + eps_z,t  with common RW trend tau_t
sig2_tau = 1.0; sig2_y = 0.5; sig2_z = 0.5;
for r = 1:R
    tau = cumsum(sqrt(sig2_tau) * randn(T,1));
    y   = tau + sqrt(sig2_y) * randn(T,1);
    z   = tau + sqrt(sig2_z) * randn(T,1);

    [b, se, tvec, R2r, u, DWr] = ols_with_const(y, z); % slope ~ 1
    b1(r)   = b(2);
    tstat(r)= tvec(2);
    R2(r)   = R2r;
    rhoU(r) = lag1corr(u);       % should be moderate (< 1)
    DW(r)   = DWr;
end
[mean_b1(4), sd_b1(4), rej_rate(4), mean_R2(4), mean_rhoU(4), mean_DW(4)] = ...
    summarize_case(b1, tstat, R2, rhoU, DW, tcrit);
t_all{4}  = tstat; R2_all{4} = R2;

%% ----------- Save a compact summary table -----------
Summary = table(caseNames.', mean_b1, sd_b1, rej_rate, mean_R2, mean_rhoU, mean_DW, ...
    'VariableNames', {'Case','mean_beta1','sd_beta1','rej_H0_at_5pct','mean_R2','mean_rho1_resid','mean_DW'});
if do_diff_fix
    Summary.rej_diff_case3 = [NaN; NaN; rej_diff; NaN];
end
writetable(Summary, fullfile(outdir,'2_spurious_regression_summary.csv'));
disp(Summary);

%% ----------- Plots: t-statistics & R^2 distributions -----------
% t-stat histograms
for k = 1:K
    fh = figure('Position', figPos); grid on; hold on
    histogram(t_all{k}, histBins, 'Normalization','pdf'); 
    xline(-tcrit, '--', '$-t_{0.975}$','LabelVerticalAlignment','bottom'); xline(tcrit, '--', '$t_{0.975}$','LabelVerticalAlignment','bottom');
    title(sprintf('t-stat of slope: %s (R=%d, T=%d)', caseNames(k), R, T));
    xlabel('$t(\hat{a}_1)$'); ylabel('Density');
    drawnow;
    exportFig(fh, sprintf('2_tstat_hist_case%d.pdf', k));
    close(fh);
end

% R^2 histograms
for k = 1:K
    fh = figure('Position', figPos); grid on; hold on
    histogram(R2_all{k}, histBins, 'Normalization','pdf'); 
    title(sprintf('$R^2$ distribution: %s (R=%d, T=%d)', caseNames(k), R, T));
    xlabel('$R^2$'); ylabel('Density');
    drawnow;
    exportFig(fh, sprintf('2_R2_hist_case%d.pdf', k));
    close(fh);
end

% Quick illustration for one replication of Case 3 (time series + scatter)
y = cumsum(randn(T,1)); z = cumsum(randn(T,1));
[b, ~, ~, ~, u] = ols_with_const(y, z);
fh = figure('Position',[100 100 880 380]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile; plot([y z]); grid on; legend('$y_t$','$z_t$','Location','best');
title('Case 3 example: two independent random walks'); xlabel('t');
nexttile; scatter(z,y,15,'filled'); grid on; hold on
xline(mean(z),'--'); yline(mean(y),'--');
title(sprintf('Scatter (\\beta_1=%.2f), residuals are I(1)', b(2)));
xlabel('$z_t$'); ylabel('$y_t$');
drawnow;
exportFig(fh,'2_case3_example_scatter_timeseries.pdf');
close(fh);

%% ===================== Exercise 3 =====================
% Invertibility, VAR(4) with Cholesky, Monte Carlo IRFs vs True IRFs

rng(20532,'twister');                         % reproducibility

% --- Settings ---
beta    = 0.6;
SigU    = diag([1, 0.8]);                     % Var[eta]=1, Var[eps]=0.8
T       = 500;                                 % sample size per simulation
B       = 200;                                 % burn-in to wash out initial lags
p       = 4;                                   % VAR order
H       = 20;                                  % IRF horizons (0..H)
Nmc     = 500;                                 % Monte Carlo replications (adjust if needed)

% --- True IRFs to unit-variance structural shocks [eta, eps/sqrt(0.8)] ---
% x_t = eta_t + L^2 eps_t
% y_t = (beta/(1-beta)) eta_t + (beta^2/(1-beta) + beta L) eps_t
% Let e2_t = eps_t / sqrt(0.8) so Var(e2)=1 -> responses to e2 have sqrt(0.8) scaling
trueIRF = zeros(2,2,H+1);           % [resp var, shock, horizon]
% Shock 1: eta_t (unit variance)
trueIRF(1,1,1) = 1;                                 % x to eta at h=0
trueIRF(2,1,1) = beta/(1-beta);                     % y to eta at h=0
% Shock 2: e2_t = eps_t/sqrt(0.8) (unit variance)
s = sqrt(0.8);
trueIRF(1,2,3) = s;                                 % x to eps at h=2
trueIRF(2,2,1) = (beta^2/(1-beta)) * s;             % y to eps at h=0
trueIRF(2,2,2) = beta * s;                           % y to eps at h=1

% --- Monte Carlo: estimate VAR(4), identify by Cholesky, store IRFs ---
IRF_draws = zeros(2,2,H+1,Nmc);      % store structural IRFs (unit-variance shocks)

for r = 1:Nmc
    % simulate data from DGP
    [X, ~] = simulate_dgp_invertibility(T+B, beta, SigU);
    X = X(B+1:end,:);  % drop burn-in
    % estimate VAR(p) with intercept
    [Acomp, Pchol] = estimate_var_chol(X, p);
    % IRFs (structural, unit-variance shocks): Theta_j = Phi_j * P
    IRF_draws(:,:,:,r) = var_irf_from_companion(Acomp, Pchol, H);
end

% --- Summaries across Monte Carlo runs ---
IRF_mean = mean(IRF_draws, 4);
IRF_lo   = prctile(IRF_draws, 2.5, 4);
IRF_hi   = prctile(IRF_draws,97.5, 4);

% --- Save results ---
save(fullfile(outdir,'ex3_invertibility_irfs.mat'), ...
     'beta','SigU','T','B','p','H','Nmc','trueIRF','IRF_draws','IRF_mean','IRF_lo','IRF_hi');

% --- Plots: True vs Estimated (mean + 95% band) ---
h = 0:H;
labelsVar   = {'$x_t$','$y_t$'};
labelsShock = {'$\eta$ (unit var)','$\varepsilon/\sqrt{0.8}$ (unit var)'};

% one figure per variable, 2 panels (shocks)
for iv = 1:2
    fh = figure('Position',[120 120 900 380]);
    tlo = tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
    for js = 1:2
        nexttile; hold on; grid on
        % bands
        fill([h, fliplr(h)], [squeeze(IRF_lo(iv,js,:))' fliplr(squeeze(IRF_hi(iv,js,:))')], ...
             [0.8 0.85 1.0], 'EdgeColor','none', 'FaceAlpha',0.6);
        % mean est
        plot(h, squeeze(IRF_mean(iv,js,:)), '-', 'LineWidth',1.8);
        % true
        plot(h, squeeze(trueIRF(iv,js,:)), '--k', 'LineWidth',1.6);
        xlabel('Horizon $h$'); ylabel('Response');
        title(sprintf('%s to %s', labelsVar{iv}, labelsShock{js}));
        legend('95\% band (MC)','MC mean','True','Location','best');
    end
    title(tlo, sprintf('Exercise 3: IRFs for %s (VAR(%d), N=%d, T=%d)', labelsVar{iv}, p, Nmc, T));
    exportFig(fh, sprintf('3_irfs_%s_VAR%d_N%d_T%d.pdf', erase(labelsVar{iv}, {'$','_','^','{','}','\'}), p, Nmc, T));
    close(fh);
end

% Combined 2x2 figure (x/y × eta/eps)
fh = figure('Position',[100 100 940 720]);
tlo = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
for iv = 1:2
    for js = 1:2
        nexttile; hold on; grid on
        fill([h, fliplr(h)], [squeeze(IRF_lo(iv,js,:))' fliplr(squeeze(IRF_hi(iv,js,:))')], ...
             [0.8 0.85 1.0], 'EdgeColor','none', 'FaceAlpha',0.6);
        plot(h, squeeze(IRF_mean(iv,js,:)), '-', 'LineWidth',1.8);
        plot(h, squeeze(trueIRF(iv,js,:)), '--k', 'LineWidth',1.6);
        xlabel('Horizon $h$'); ylabel('Response');
        title(sprintf('%s to %s', labelsVar{iv}, labelsShock{js}));
        if iv==1 && js==1
            legend('95\% band (MC)','MC mean','True','Location','best');
        end
    end
end
title(tlo, sprintf('Exercise 3: IRFs (VAR(%d), N=%d, T=%d), Cholesky', p, Nmc, T));
exportFig(fh, sprintf('3_irfs_all_VAR%d_N%d_T%d.pdf', p, Nmc, T));
close(fh);

%% ===================== Exercise 4 =====================
% Romer & Romer (2004): VAR(4) and Granger-causality with R&R shocks

% -------- Parameters --------
p = 4;                                 % VAR lag order (assignment requires 4)
romer_path = '/Users/stefanograziosi/Documents/GitHub/20532-macroeconometrics-ps/ps2/Data/Romer_Romer2004.csv';     % Adjust path if needed
alpha = 0.05;                           % test size for printing

Ttbl = readtable(romer_path, 'PreserveVariableNames', true);
Ttbl.Properties.VariableNames = lower(Ttbl.Properties.VariableNames);      % Normalize variable names to lowercase for robust indexing

% Expected columns in this order
varOrder = {'inflation','unemployment','ffr','rr_shock'};
assert(all(ismember(varOrder, Ttbl.Properties.VariableNames)), ...
    'CSV must contain columns: %s', strjoin(varOrder, ', '));

Yraw = double(Ttbl{:, varOrder});      % [Infl, Unemp, FFR, RR_Shock]

% Drop any rows with NaNs
ok = all(~isnan(Yraw), 2);
Yraw = Yraw(ok, :);
[Tobs, n] = size(Yraw);

% -------- Build design matrix: [const | L1 all vars | L2 all vars | ...] --------
% Use your mlag helper: it returns exactly [L1 all vars | L2 all vars | ...]
Xlags = mlag(Yraw, p);                 % first p rows contain zeros by design
X = [ones(Tobs,1) Xlags];
Y = Yraw;

% Trim first p rows to align lags
Xt = X(p+1:end, :);
Yt = Y(p+1:end, :);
[T, K] = size(Xt);                     % T effective obs, K regressors

df = T - K;                            % residual df per equation

% -------- OLS by equation (equation-by-equation OLS is efficient for VAR) --------
XX = Xt' * Xt;                          % K x K
% Use backslash for both the estimator and (XX)^{-1}
B = XX \ (Xt' * Yt);                   % K x n, column j are coeffs for eqn j
U = Yt - Xt * B;                        % T x n residuals
s2 = sum(U.^2, 1) ./ df;                % 1 x n residual variances (per equation)

% Compute (X'X)^{-1} stably without explicit inv
invXX = XX \ eye(K);                   % numerically safer than inv(XX)

% -------- Helper to map lag-block positions --------
% With our X layout, positions for variable v (1..n) across all p lags are:
% posLag(v) = 1 + [0, n, 2n, ..., (p-1)n] + v
posLag = @(v) 1 + ( (0:p-1)' * n + v );  % column vector of length p

% Labels for pretty output
labels = ["Inflation","Unemployment","FedFundsRate","RR_Shock"];
shockIdx = 4;

% -------- WALD tests --------
rows = struct('Cause', [], 'Arrow', [], 'Effect', [], 'NumLags', [], ...
              'NumRestrictions', [], 'Wald_chi2', [], 'p_chi2', [], ...
              'Fstat', [], 'p_F', []);
res = repmat(rows, 7, 1);   % 3 A-tests + 3 B-tests + 1 C-test = 7 rows
rptr = 0;

% (A) Do R&R shocks Granger-cause the others?  (Shock -> yEq for yEq=1..3)
for yEq = 1:3
    rptr = rptr + 1;
    bj = B(:, yEq);
    s2j = s2(yEq);
    cols = posLag(shockIdx);
    R = zeros(p, K);
    for r = 1:p, R(r, cols(r)) = 1; end
    [W, pchi2, Fstat, pF] = waldTest_ols(bj, s2j, invXX, R, df);
    res(rptr) = packrow(labels(shockIdx), "→", labels(yEq), p, size(R,1), W, pchi2, Fstat, pF);
end

% (B) Do others Granger-cause the R&R shocks?  (xVar -> Shock) individually
for xVar = 1:3
    rptr = rptr + 1;
    bj = B(:, shockIdx);
    s2j = s2(shockIdx);
    cols = posLag(xVar);
    R = zeros(p, K);
    for r = 1:p, R(r, cols(r)) = 1; end
    [W, pchi2, Fstat, pF] = waldTest_ols(bj, s2j, invXX, R, df);
    res(rptr) = packrow(labels(xVar), "→", labels(shockIdx), p, size(R,1), W, pchi2, Fstat, pF);
end

% (C) Others jointly -> Shock  (r = 3p)
rptr = rptr + 1;
bj = B(:, shockIdx);
s2j = s2(shockIdx);
R = zeros(3*p, K);
row = 0;
for xVar = 1:3
    cols = posLag(xVar);
    for r = 1:p
        row = row + 1;
        R(row, cols(r)) = 1;
    end
end
[W, pchi2, Fstat, pF] = waldTest_ols(bj, s2j, invXX, R, df);
res(rptr) = packrow("All{Infl,Unemp,FFR}", "→", labels(shockIdx), p, size(R,1), W, pchi2, Fstat, pF);

% -------- Pretty print & save --------
ResultsTbl = struct2table(res);

fprintf('\n=== Exercise 4: Granger-causality (VAR(%d), levels, intercept only) ===\n', p);
disp(ResultsTbl);

% Save next to the data file
outdir = fileparts(romer_path); if isempty(outdir), outdir = pwd; end
outcsv = fullfile(outdir,'4_romer_granger_results.csv');
writetable(ResultsTbl, outcsv);
fprintf('\nSaved results to: %s\n', outcsv);

% Quick console interpretation
fprintf('\nAt alpha = %.2f, reject H0 (no Granger-causality) when p-values < %.2f.\n', alpha, alpha);
for i = 1:height(ResultsTbl)
    rej = ResultsTbl.p_F(i) < alpha;
    tag = ternary(rej, 'REJECT', 'do not reject'); % your helper
    fprintf('%-18s %s %-15s : p_F = %.4f  -> %s\n', ...
        char(ResultsTbl.Cause(i)), char(ResultsTbl.Arrow(i)), char(ResultsTbl.Effect(i)), ResultsTbl.p_F(i), tag);
end

%% ========================= Exercise 5 =========================

rng(20532,'twister');                       % reproducibility (safe to repeat)

% ---- Settings ----
T5       = 250;                             % sample length
R5       = 5000;                            % Monte Carlo replications
sigma2_5 = 0.6;                             % innovation variance
alpha0_5 = 0.5;                             % intercept in the DGP (levels)
beta1_5  = 0.05;                            % trend slope for part (b)

% Critical values
z975 = 1.95996398454005;                    % two-sided 5% Normal
df_a = (T5-1) - 2;                          % (a) levels + intercept
df_b = (T5-1) - 3;                          % (b) levels + intercept + trend
if exist('tinv','file')
    t975_a = tinv(0.975, df_a);
    t975_b = tinv(0.975, df_b);
else
    t975_a = z975;                          % fallback to Normal if tinv unavailable
    t975_b = z975;
end

%%% Question (a)
% What happens to the distribution of the t-test in this case? How do you intuitively explain this? Do you notice anything interesting on the mean and on the shape of the distribution of the t-statistic?
% DGP: y_t = alpha0_5 + y_{t-1} + eps_t
% OLS: y_t = a + rho*y_{t-1} + e_t, test H0: rho = 1

t_rho_a      = zeros(R5,1);     % t-stat for H0: rho=1
rho_hat_a    = zeros(R5,1);
alpha_hat_a  = zeros(R5,1);
t_alpha_a    = zeros(R5,1);
se_rho_a     = zeros(R5,1);

for r = 1:R5
    % --- simulate ---
    eps = sqrt(sigma2_5) * randn(T5,1);
    y   = zeros(T5,1);
    for t = 2:T5
        y(t) = alpha0_5 + y(t-1) + eps(t);
    end

    % --- OLS: y_t on [1, y_{t-1}] ---
    yt   = y(2:end);
    xlag = y(1:end-1);
    X    = [ones(T5-1,1), xlag];
    XX   = X' * X;
    b    = XX \ (X' * yt);
    u    = yt - X * b;
    s2   = (u' * u) / df_a;
    invXX= XX \ eye(2);
    Vb   = s2 * invXX;
    se   = sqrt(diag(Vb));

    alpha_hat_a(r) = b(1);
    rho_hat_a(r)   = b(2);
    se_rho_a(r)    = se(2);
    t_alpha_a(r)   = b(1) / se(1);
    t_rho_a(r)     = (b(2) - 1) / se(2);
end

% Rejection rates (two-sided 5%)
rejN_2s_a = mean(abs(t_rho_a) > z975);
rejT_2s_a = mean(abs(t_rho_a) > t975_a);

% Save raw and summary
Results5a = table(rho_hat_a, se_rho_a, t_rho_a, alpha_hat_a, t_alpha_a, ...
    'VariableNames', {'rho_hat','se_rho','t_rho_H0eq1','alpha_hat','t_alpha'});
writetable(Results5a, fullfile(outdir,'5a_levels_intercept_raw.csv'));

Summ5a = table( ...
    mean(rho_hat_a), std(rho_hat_a), mean(t_rho_a), std(t_rho_a), ...
    skewness(t_rho_a), kurtosis(t_rho_a), rejN_2s_a, rejT_2s_a, ...
    'VariableNames', {'mean_rho','sd_rho','mean_t','sd_t','skew_t','kurt_t','rej_norm_5pct_2s','rej_t_5pct_2s'});
writetable(Summ5a, fullfile(outdir,'5a_levels_intercept_summary.csv'));

% Plots
plot_t_hist_with_normal(t_rho_a, z975, ...
    '(5a) $t$-stat for $H_0:\ \rho=1$ (levels, intercept only)', ...
    exportFig, '5a_tstat_hist_levels_intercept.pdf');

normal_qqplot_simple(t_rho_a, ...
    '(5a) Normal QQ-plot of $t$-stat ($H_0:\ \rho=1$)', ...
    exportFig, '5a_tstat_normal_qq.pdf');


%%% Question (b)
% Based on what we have learnt at point a., add a time trend (another deterministic regressor) to both the DGP and the estimating equation at point a. and check that also in this case the results of Sims, Stock and Watson hold [summarized in Rule 2 in Enders, p.267].
% DGP: y_t = alpha0_5 + beta1_5*t + y_{t-1} + eps_t
% OLS: y_t = a + b*t + rho*y_{t-1} + e_t, test H0: rho = 1

t_rho_b    = zeros(R5,1);
rho_hat_b  = zeros(R5,1);
se_rho_b   = zeros(R5,1);
a_hat_b    = zeros(R5,1);
b_hat_b    = zeros(R5,1);
t_a_b      = zeros(R5,1);
t_b_b      = zeros(R5,1);

tvec5 = (1:T5)';

for r = 1:R5
    % --- simulate with trend ---
    eps = sqrt(sigma2_5) * randn(T5,1);
    y   = zeros(T5,1);
    for t = 2:T5
        y(t) = alpha0_5 + beta1_5*t + y(t-1) + eps(t);
    end

    % --- OLS: y_t on [1, t, y_{t-1}] ---
    yt   = y(2:end);
    xlag = y(1:end-1);
    tt   = tvec5(2:end);
    X    = [ones(T5-1,1), tt, xlag];
    XX   = X' * X;
    b    = XX \ (X' * yt);
    u    = yt - X * b;
    s2   = (u' * u) / df_b;
    invXX= XX \ eye(3);
    Vb   = s2 * invXX;
    se   = sqrt(diag(Vb));

    a_hat_b(r)   = b(1);
    b_hat_b(r)   = b(2);
    rho_hat_b(r) = b(3);
    se_rho_b(r)  = se(3);

    t_a_b(r)     = b(1) / se(1);
    t_b_b(r)     = b(2) / se(2);
    t_rho_b(r)   = (b(3) - 1) / se(3);
end

% Rejection rates (two-sided 5%)
rejN_2s_b = mean(abs(t_rho_b) > z975);
rejT_2s_b = mean(abs(t_rho_b) > t975_b);

% Save raw and summary
Results5b = table(rho_hat_b, se_rho_b, t_rho_b, a_hat_b, t_a_b, b_hat_b, t_b_b, ...
    'VariableNames', {'rho_hat','se_rho','t_rho_H0eq1','alpha_hat','t_alpha','beta_hat','t_beta'});
writetable(Results5b, fullfile(outdir,'5b_levels_trend_raw.csv'));

Summ5b = table( ...
    mean(rho_hat_b), std(rho_hat_b), mean(t_rho_b), std(t_rho_b), ...
    skewness(t_rho_b), kurtosis(t_rho_b), rejN_2s_b, rejT_2s_b, ...
    'VariableNames', {'mean_rho','sd_rho','mean_t','sd_t','skew_t','kurt_t','rej_norm_5pct_2s','rej_t_5pct_2s'});
writetable(Summ5b, fullfile(outdir,'5b_levels_trend_summary.csv'));

% Plots
plot_t_hist_with_normal(t_rho_b, z975, ...
    '(5b) $t$-stat for $H_0:\ \rho=1$ (levels, intercept + trend)', ...
    exportFig, '5b_tstat_hist_levels_trend.pdf');

normal_qqplot_simple(t_rho_b, ...
    '(5b) Normal QQ-plot of $t$-stat ($H_0:\ \rho=1$) with trend', ...
    exportFig, '5b_tstat_normal_qq.pdf');

%% ---------- Compact console summary & percentiles ----------
fprintf('\n=== Exercise 5(a): Intercept only ===\n');
fprintf('mean(t) = %.3f, sd(t) = %.3f, skew = %.3f, kurt = %.3f\n', ...
    mean(t_rho_a), std(t_rho_a), skewness(t_rho_a), kurtosis(t_rho_a));
fprintf('Rej@5%% (two-sided): Normal = %.3f, Student(df=%d) = %.3f\n', ...
    rejN_2s_a, df_a, rejT_2s_a);

fprintf('\n=== Exercise 5(b): Intercept + trend ===\n');
fprintf('mean(t) = %.3f, sd(t) = %.3f, skew = %.3f, kurt = %.3f\n', ...
    mean(t_rho_b), std(t_rho_b), skewness(t_rho_b), kurtosis(t_rho_b));
fprintf('Rej@5%% (two-sided): Normal = %.3f, Student(df=%d) = %.3f\n', ...
    rejN_2s_b, df_b, rejT_2s_b);

pct_vec5 = [1 5 10 25 50 75 90 95 99]';
PTa = prctile(t_rho_a, pct_vec5);
PTb = prctile(t_rho_b, pct_vec5);
writetable(table(pct_vec5, PTa, PTb, ...
    'VariableNames',{'percentile','t_levels_intercept_only','t_levels_trend'}), ...
    fullfile(outdir,'5_tstat_percentiles.csv'));

%% ------------------------ Functions used --------------------------
function Y = simulate_ar1_loop(T, phi, sigma2, mu, Y0)
    if nargin < 5, Y0 = mu; end
    eps = sqrt(sigma2) * randn(T,1);
    Y       = zeros(T,1);
    Y(1)    = mu + phi*(Y0 - mu) + eps(1);
    for t = 2:T
        Y(t) = mu + phi*(Y(t-1) - mu) + eps(t);
    end
end

function [b, se, tstat, R2, u, DW] = ols_with_const(y, z)
    % OLS of y on [1 z], standard errors, t-stats, R^2, residuals, DW
    T = length(y);
    X = [ones(T,1), z(:)];
    XX = X' * X;
    b  = XX \ (X' * y);
    u  = y - X * b;
    k  = size(X,2);
    s2 = (u' * u) / (T - k);
    V  = s2 * inv(XX);
    se = sqrt(diag(V));
    tstat = b ./ se;
    R2 = 1 - (u' * u) / sum( (y - mean(y)).^2 );
    DU = diff(u);
    DW = sum(DU.^2) / (u' * u);
end

function r1 = lag1corr(u)
    u1 = u(1:end-1); u2 = u(2:end);
    r1 = ( (u1 - mean(u1))' * (u2 - mean(u2)) ) / ( std(u1) * std(u2) * (numel(u1)-1) );
end

function [m_b, sd_b, rej, m_R2, m_rho, m_DW] = summarize_case(b1, tstat, R2, rhoU, DW, tcrit)
    m_b   = mean(b1);
    sd_b  = std(b1);
    rej   = mean(abs(tstat) > tcrit);
    m_R2  = mean(R2);
    m_rho = mean(rhoU);
    m_DW  = mean(DW);
end

% For Exercise 3

function [XY, Y_only] = simulate_dgp_invertibility(T, beta, SigU)
% Simulate from:
% [x_t; y_t] = [1, L^2; beta/(1-beta), beta^2/(1-beta) + beta L] * [eta_t; eps_t]
% with Var(eta)=SigU(1,1), Var(eps)=SigU(2,2), uncorrelated.
    s_eta = sqrt(SigU(1,1));
    s_eps = sqrt(SigU(2,2));
    eta = s_eta * randn(T,1);
    eps = s_eps * randn(T,1);
    x = zeros(T,1);
    y = zeros(T,1);
    for t=1:T
        % x_t = eta_t + eps_{t-2}
        x(t) = eta(t);
        if t-2 >= 1
            x(t) = x(t) + eps(t-2);
        end
        % y_t = (beta/(1-beta)) eta_t + (beta^2/(1-beta)) eps_t + beta eps_{t-1}
        y(t) = beta/(1-beta) * eta(t) + (beta^2/(1-beta)) * eps(t);
        if t-1 >= 1
            y(t) = y(t) + beta * eps(t-1);
        end
    end
    XY = [x y];
    if nargout>1, Y_only = y; end
end

function [Acomp, P] = estimate_var_chol(Y, p)
% Estimate VAR(p) with intercept by OLS equation-by-equation.
% Return companion matrix Acomp (size np x np) and Cholesky P of residual Σ_u (lower).
    [T, n] = size(Y);
    Xlag = mlag(Y, p);                 % T x (n*p)
    X = [ones(T,1), Xlag];             % intercept
    Ytrim = Y(p+1:end, :);
    Xtrim = X(p+1:end, :);
    B = (Xtrim' * Xtrim) \ (Xtrim' * Ytrim);   % (1+n*p) x n
    U = Ytrim - Xtrim * B;                      % T-p x n residuals
    SigmaU = (U' * U) / (size(U,1) - (1 + n*p));
    % Companion
    A = B(2:end,:).';                            % n x (n*p) (stacked [A1 ... Ap])
    A1toP = A;
    Acomp = zeros(n*p, n*p);
    Acomp(1:n, :) = A1toP;
    if p>1
        Acomp(n+1:end, 1:n*(p-1)) = eye(n*(p-1));
    end
    % Cholesky (lower) for Σ_u
    P = chol(SigmaU, 'lower');
end

function Xlags = mlag(X, p)
% Create lag matrix [L1 .. Lp] for each column of X. Missing top rows left as zeros.
    [T, n] = size(X);
    Xlags = zeros(T, n*p);
    for k=1:p
        Xlags(k+1:end, (n*(k-1)+1):n*k) = X(1:end-k, :);
    end
end

function TH = var_irf_from_companion(Acomp, P, H)
% Structural IRFs Theta_h = Phi_h * P, where Phi_h from companion powers.
    np = size(Acomp,1);
    n  = size(P,1);
    J  = [eye(n), zeros(n, np-n)];
    TH = zeros(n,n,H+1);
    Phi = eye(n);                 % Phi_0
    TH(:,:,1) = Phi * P;
    A_pow = eye(np);
    for h = 1:H
        A_pow = A_pow * Acomp;
        Phi_h = J * A_pow * J';
        TH(:,:,h+1) = Phi_h * P;
    end
end

% For Exercise 4

function o = ternary(cond, a, b), if cond, o = a; else, o = b; end, end

function [W, pchi2, Fstat, pF] = waldTest_ols(bj, s2j, invXX, R, df)
% Classic Wald test under homoskedastic OLS for a *single* equation j
% H0: R * b_j = 0
r = size(R,1);
Rb = R * bj;
Vb = s2j * (R * invXX * R');          % var of Rb under OLS homoskedasticity
% Use a solve for numerical stability (no explicit inv)
W = Rb' / Vb * Rb;                     % equivalent to (Rb' * inv(Vb) * Rb)
Fstat = W / r;                         % F(r, df)
pF = 1 - fcdf(Fstat, r, df);
pchi2 = 1 - chi2cdf(W, r);
end

function row = packrow(cause, arrow, effect, p, nr, W, pchi2, F, pF)
row = struct('Cause', string(cause), 'Arrow', string(arrow), 'Effect', string(effect), ...
             'NumLags', p, 'NumRestrictions', nr, 'Wald_chi2', W, 'p_chi2', pchi2, ...
             'Fstat', F, 'p_F', pF);
end

% For Exercise 5

function plot_t_hist_with_normal(tstats, zcrit, ttl, exportFig, outname)
% Plot histogram of t-stats with N(0,1) overlay and ±zcrit lines, then save via exportFig
    fh = figure('Position', [100 100 860 420]); hold on; grid on
    histogram(tstats, 55, 'Normalization','pdf');
    xx = linspace(min(tstats), max(tstats), 400);
    nn = normal_pdf(xx);
    plot(xx, nn, '-', 'LineWidth', 1.6);
    xline(-zcrit, '--', '$-z_{0.975}$', 'LabelVerticalAlignment','bottom');
    xline( zcrit, '--', '$z_{0.975}$',  'LabelVerticalAlignment','bottom');
    title(ttl);
    xlabel('$t = (\hat{\rho}-1)/\text{se}(\hat{\rho})$'); ylabel('Density');
    legend('Empirical','Normal(0,1)','Location','best');
    if nargin >= 5 && ~isempty(outname)
        exportFig(fh, outname);
        close(fh);
    end
end

function normal_qqplot_simple(tstats, ttl, exportFig, outname)
% Toolbox-free Normal QQ-plot using erfinv-based Normal quantiles
    R = numel(tstats);
    q = linspace(0.5/R, 1-0.5/R, 200)';       % interior quantiles for stability
    emp = quantile(tstats, q);
    the = normal_icdf(q);
    fh = figure('Position', [100 100 860 420]); hold on; grid on
    scatter(the, emp, 12, 'filled');
    plot(the, the, 'k--', 'LineWidth', 1.2);
    xlabel('Theoretical Normal quantiles'); ylabel('Empirical quantiles');
    title(ttl);
    if nargin >= 4 && ~isempty(outname)
        exportFig(fh, outname);
        close(fh);
    end
end

function y = normal_pdf(x)
% PDF of N(0,1) without Statistics Toolbox
    y = (1./sqrt(2*pi)) .* exp(-0.5 .* (x.^2));
end

function x = normal_icdf(p)
% Inverse CDF of N(0,1) using erfinv (base MATLAB)
% Valid for p in (0,1); vectorized.
    x = sqrt(2) .* erfinv(2.*p - 1);
end