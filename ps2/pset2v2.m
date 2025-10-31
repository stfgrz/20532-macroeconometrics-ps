%% 20532 Macroeconometrics | Problem Set 2
%
% ---------------------------------------------------------------
% Author: Stefano Graziosi
% Date: 2025-10-31
% ---------------------------------------------------------------

%% Housekeeping & Graphics Style
clear; clc; close all; format compact;

% Output directory setup
outdir = fullfile(pwd,'ps2/output');
if ~exist(outdir,'dir'), mkdir(outdir); end

% Clean, consistent figure defaults
set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLineLineWidth', 1.5);
set(groot, 'defaultAxesToolbarVisible', 'off');

% Helper to export figures
exportFig = @(fh, name) exportgraphics(fh, fullfile(outdir, name), 'ContentType','vector');

% Global settings & Critical Values
rng(20532,'twister');                               % Set seed for reproducibility
z_5_left = -1.6448536269;                           % 5% one-sided Normal critical value (left tail)
chi2_95_df1 = 3.8414588207;                         % 95% Chi-squared critical value with 1 df

% Dickey-Fuller critical values for T~250
DFcrit = struct();
DFcrit.tau_mu   = struct('p1',-3.46,'p5',-2.88,'p10',-2.57); % Intercept only
DFcrit.tau_trend= struct('p1',-3.99,'p5',-3.43,'p10',-3.12); % Intercept and trend

%% Exercise 1

% Monte Carlo settings
T        = 250;              % Sample length
R        = 5000;             % Monte Carlo replications
sigma2   = 0.6;              % Variance of innovations

%%% Question (a):
%   Compute the empirical distribution of the OLS estimator in the case of an AR(1) with $\varphi = 1$ and $T=250$ (you are free to choose the variance of the innovation).

phi_hat_a = zeros(R,1);
for r = 1:R
    y = simulate_ar1(T, 1, sigma2, 0, 0); % Pure random walk (phi=1)
    stats = run_ols(y(2:end), y(1:end-1), false); % No constant
    phi_hat_a(r) = stats.b;
end

% Plot histogram
figPos = [100 100 840 420];
histBins = 50;
fh_a = figure('Position', figPos);
histogram(phi_hat_a, histBins, 'Normalization','pdf'); grid on; hold on;
xline(1,'--','True $\phi=1$','LabelVerticalAlignment','bottom');
xlabel('$\hat{\phi}$'); ylabel('Density');
title('(a) Empirical distribution of OLS $\hat{\phi}$ under unit root ($T=250$)');
exportFig(fh_a,'1a_phi_hat_hist.pdf');
close(fh_a);

fprintf('Q1(a) mean(\\hat{phi})=%.4f, sd=%.4f, med=%.4f\n', mean(phi_hat_a), std(phi_hat_a), median(phi_hat_a));

%%% Question (b):
%   Repeat the exercise in (a) but now with a drift term equal to Construct a $t$-test for the null hypothesis $H_{0}:\ \rho=\varphi-1=0$, in a test regression: $\Delta y_{t}=\alpha+\rho y_{t-1}+\varepsilon_{t}$; against a one-sided alternative $H_{0}:\ \rho<0$.
%   Using a standard Normal distribution, how often do you reject the null hypothesis at the $95\%$ confidence level?
%   Is the actual distribution of the t-test symmetric? Discuss.

t_b   = zeros(R,1);
rho_b = zeros(R,1);

for r = 1:R
    y = simulate_ar1(T, 1, sigma2, 0, 0); % Pure random walk
    dy = diff(y);
    ylag = y(1:end-1);
    stats = run_ols(dy, ylag, true); % Regression with constant
    rho_b(r) = stats.b(2);
    t_b(r)   = stats.tstat(2);
end

rej_norm_left = mean(t_b < z_5_left);
sk_b = skewness(t_b);
fprintf(['(b) Using Normal 5%% one-sided (z=%.3f): reject rate = %.3f. ', ...
         'Skewness of t-stat = %.3f (non-symmetric).\n'], z_5_left, rej_norm_left, sk_b);

% Plot histogram of t-statistic
fh_b = figure('Position', figPos);
histogram(t_b, histBins, 'Normalization','pdf'); grid on; hold on;
xline(z_5_left, '--', 'Normal 5% one-sided crit','LabelVerticalAlignment','bottom');
xlabel('$t(\hat{\rho})$'); ylabel('Density');
title('(b) DF $t$-stat under unit root, intercept included');
exportFig(fh_b,'1b_t_hist.pdf');
close(fh_b);

%%% Question (c):
%   Compute now few percentiles of the empirical distribution of the $t$-test you generated at point b. and check that they are close to those tabulated by Dickey and Fuller.

pct_vec = [1 5 10 25 50 75 90 95 99];
emp_pct = prctile(t_b, pct_vec);

% Print a compact comparison
fprintf('(c) Empirical DF t percentiles (%%):\n');
disp(table(pct_vec(:), emp_pct(:), 'VariableNames',{'percentile','empirical_t'}));
fprintf('Tabulated DF (tau_mu) ~ T=250: 1%%=%.2f, 5%%=%.2f, 10%%=%.2f\n', ...
        DFcrit.tau_mu.p1, DFcrit.tau_mu.p5, DFcrit.tau_mu.p10);

% Compare left-tail critical values
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

writetable(comp_tbl, fullfile(outdir,'1c_empirical_vs_DF_tau_mu.csv'));

%%% Question (d)
% Compute the empirical distribution of the OLS in the case of a random walk with drift and $T = 250$ and study the performance of the Dickey--Fuller test.

a0_drift = 0.5;
phi_hat_d = zeros(R,1);
t_d = zeros(R,1);
for r = 1:R
    y = simulate_ar1(T, 1, sigma2, a0_drift, 0); % Random walk with drift
    
    % OLS of y_t on [1, y_{t-1}]
    stats_level = run_ols(y(2:end), y(1:end-1), true);
    phi_hat_d(r) = stats_level.b(2);

    % Dickey–Fuller regression
    dy = diff(y);
    stats_df = run_ols(dy, y(1:end-1), true);
    t_d(r) = stats_df.tstat(2);
end

rej_norm_d = mean(t_d < z_5_left);
rej_DF5_d  = mean(t_d < DFcrit.tau_mu.p5);
fprintf(['(d) OLS under RW+drift: mean(phi-hat)=%.4f, sd=%.4f.\n' ...
         '    DF (one-sided) reject@5%% using Normal: %.3f; using DF τ_μ: %.3f\n'], ...
        mean(phi_hat_d), std(phi_hat_d), rej_norm_d, rej_DF5_d);

%%% Question(e)
%Construct an F-test for the null hypothesis $H_{0}$: there is unit root, against the alternative $H_{1}$: there is no unit root using a $\chi^{2}$ distribution (how many degrees of freedom?). How often do you reject $H_{0}$ at 95\% confidence?

chi2_stat = t_d.^2;
rej_chi2  = mean(chi2_stat > chi2_95_df1);
fprintf('(e) Wald χ² test (df=1) reject@95%% under H0 (RW+drift): %.3f\n', rej_chi2);

% Visualize t and χ^2 stats
fh_e1 = figure('Position', figPos);
histogram(t_d, histBins, 'Normalization','pdf'); grid on; hold on
xline(-sqrt(chi2_95_df1),'--','$\pm \sqrt{\chi^2_{0.95;1}}$','LabelVerticalAlignment','bottom');
xline(+sqrt(chi2_95_df1),'--');
xlabel('$t(\hat{\rho})$'); ylabel('Density'); title('(e) DF $t$-stats under $H_0$ (RW+drift)')
exportFig(fh_e1,'1e_t_hist_RWdrift.pdf'); close(fh_e1);

%%% Question (f)
% Generate now data from a deterministic time trend and perform a DF test using the correct distribution for the test with null hypothesis $H_{0}$: there is unit root. How often do you reject the null? \emph{(hint: you can find additional details in Enders).}

beta0 = 0.0;
beta1 = 0.05;
tvec   = (1:T)';
t_f = zeros(R,1);
for r = 1:R
    eps = sqrt(sigma2)*randn(T,1);
    y   = beta0 + beta1*tvec + eps; % Trend-stationary process
    dy  = diff(y);
    ylag= y(1:end-1);
    t2  = tvec(2:end);
    Xtr = [t2, ylag]; % Regressors for DF test with trend
    stats = run_ols(dy, Xtr, true); % run_ols adds constant
    t_f(r)  = stats.tstat(3); % t-stat for y_lag
end

rej_trend_DF5 = mean(t_f < DFcrit.tau_trend.p5);
fprintf('(f) Trend-stationary DGP: DF with trend (τ_τ) reject@5%% = %.3f (power against unit root).\n', rej_trend_DF5);

%% Exercise 2

% Monte Carlo settings
T2   = 250;
R2   = 2000;
B2   = 500; % Burn-in
nu  = T2 - 2;
tcrit = tinv(0.975, nu);

% Case setup
caseNames = ["Case 1: both stationary", ...
             "Case 2: I(1) vs I(0)", ...
             "Case 3: I(1) vs I(1) (spurious)", ...
             "Case 4: I(1) \& cointegrated"];
K = numel(caseNames);

% Result containers
results = cell(K,1);
[t_all, R2_all] = deal(cell(K,1));

%%% Case 1: Stationary vs Stationary
% y_t = 0.5 y_{t-1} + e^y_t,  z_t = -0.3 z_{t-1} + e^z_t  (independent)

[b1, t, r2, dw] = deal(zeros(R2,1));
for r = 1:R2
    y = simulate_ar1(T2+B2, 0.5, 1.0, 0, 0);
    z = simulate_ar1(T2+B2, -0.3, 1.0, 0, 0);
    stats = run_ols(y(B2+1:end), z(B2+1:end), true);
    b1(r) = stats.b(2); t(r) = stats.tstat(2); r2(r) = stats.R2; dw(r) = stats.DW;
end
results{1} = summarize_case(b1, t, r2, dw, tcrit);
t_all{1} = t; R2_all{1} = r2;

%%% Case 2: I(1) vs I(0)
% y_t is random walk; z_t stationary AR(1)

for r = 1:R2
    y = simulate_ar1(T2, 1, 1.0, 0, 0);
    z = simulate_ar1(T2+B2, 0.6, 1.0, 0, 0);
    stats = run_ols(y, z(B2+1:end), true);
    b1(r) = stats.b(2); t(r) = stats.tstat(2); r2(r) = stats.R2; dw(r) = stats.DW;
end
results{2} = summarize_case(b1, t, r2, dw, tcrit);
t_all{2} = t; R2_all{2} = r2;

%%% Case 3: I(1) vs I(1) (Spurious)

for r = 1:R2
    y = simulate_ar1(T2, 1, 1.0, 0, 0);
    z = simulate_ar1(T2, 1, 1.0, 0, 0);
    stats = run_ols(y, z, true);
    b1(r) = stats.b(2); t(r) = stats.tstat(2); r2(r) = stats.R2; dw(r) = stats.DW;
end
results{3} = summarize_case(b1, t, r2, dw, tcrit);
t_all{3} = t; R2_all{3} = r2;

% --- Case 4: Cointegrated I(1) ---
sig2_tau = 1.0; sig2_y = 0.5; sig2_z = 0.5;
for r = 1:R2
    tau = cumsum(sqrt(sig2_tau) * randn(T2,1));
    y   = tau + sqrt(sig2_y) * randn(T2,1);
    z   = tau + sqrt(sig2_z) * randn(T2,1);
    stats = run_ols(y, z, true);
    b1(r) = stats.b(2); t(r) = stats.tstat(2); r2(r) = stats.R2; dw(r) = stats.DW;
end
results{4} = summarize_case(b1, t, r2, dw, tcrit);
t_all{4} = t; R2_all{4} = r2;

% --- Summarize and Save Results ---
Summary = struct2table(vertcat(results{:}));
Summary.Case = caseNames.';
Summary = Summary(:, [width(Summary) 1:width(Summary)-1]); % Move Case to first column
writetable(Summary, fullfile(outdir,'2_spurious_regression_summary.csv'));
disp(Summary);

% --- Plots for Exercise 2 ---
for k = 1:K
    fh = figure('Position', figPos); grid on; hold on;
    histogram(t_all{k}, histBins, 'Normalization','pdf');
    xline([-tcrit, tcrit], '--', {'$-t_{0.975}$', '$t_{0.975}$'},'LabelVerticalAlignment','bottom');
    title(sprintf('t-stat of slope: %s', caseNames(k)));
    exportFig(fh, sprintf('2_tstat_hist_case%d.pdf', k));
    close(fh);

    fh = figure('Position', figPos); grid on; hold on;
    histogram(R2_all{k}, histBins, 'Normalization','pdf');
    title(sprintf('$R^2$ distribution: %s', caseNames(k)));
    exportFig(fh, sprintf('2_R2_hist_case%d.pdf', k));
    close(fh);
end

%% ===================== Exercise 3 =====================
% Invertibility, VAR(4) with Cholesky, Monte Carlo IRFs vs True IRFs

% --- Settings ---
beta3    = 0.6;
SigU3    = diag([1, 0.8]);
T3       = 500;
B3       = 200;
p3       = 4;
H3       = 20;
Nmc3     = 500;

% --- True IRFs ---
trueIRF = compute_true_irfs_ex3(beta3, SigU3, H3);

% --- Monte Carlo ---
IRF_draws = zeros(2,2,H3+1,Nmc3);
for r = 1:Nmc3
    [X, ~] = simulate_dgp_invertibility(T3+B3, beta3, SigU3);
    X = X(B3+1:end,:);
    [Acomp, Pchol] = estimate_var_chol(X, p3);
    IRF_draws(:,:,:,r) = var_irf_from_companion(Acomp, Pchol, H3);
end

% --- Summaries and Plots ---
IRF_mean = mean(IRF_draws, 4);
IRF_lo   = prctile(IRF_draws, 2.5, 4);
IRF_hi   = prctile(IRF_draws, 97.5, 4);

plot_irfs_ex3(trueIRF, IRF_mean, IRF_lo, IRF_hi, p3, Nmc3, T3, exportFig);

%% ===================== Exercise 4 =====================
% Romer & Romer (2004): VAR(4) and Granger-causality

% --- Parameters & Data Loading ---
p4 = 4;
romer_path = fullfile(pwd, 'ps2', 'data', 'Romer_Romer2004.csv');
alpha4 = 0.05;

Ttbl = readtable(romer_path, 'PreserveVariableNames', true);
Ttbl.Properties.VariableNames = lower(Ttbl.Properties.VariableNames);

varOrder = {'inflation','unemployment','ffr','rrshock'};
Yraw = double(Ttbl{:, varOrder});
Yraw = Yraw(all(~isnan(Yraw), 2), :);
[Tobs, n] = size(Yraw);

% --- VAR Estimation ---
Xlags = mlag(Yraw, p4);
X = [ones(Tobs,1), Xlags];
Y = Yraw;

Xt = X(p4+1:end, :);
Yt = Y(p4+1:end, :);
[T_eff, K] = size(Xt);
df = T_eff - K;

B = Xt \ Yt;
U = Yt - Xt * B;
s2 = sum(U.^2, 1) ./ df;
invXX = (Xt' * Xt) \ eye(K);

% --- Granger-Causality Tests ---
labels = ["Inflation","Unemployment","FedFundsRate","rrshock"];
shockIdx = 4;

res = [];
% (A) Shock -> Others
for yEq = 1:3
    res = [res; run_gc_test(B, s2, invXX, df, p4, K, shockIdx, yEq, labels)]; 
end
% (B) Others -> Shock (individually)
for xVar = 1:3
    res = [res; run_gc_test(B, s2, invXX, df, p4, K, xVar, shockIdx, labels)];
end
% (C) Others -> Shock (jointly)
res = [res; run_gc_test(B, s2, invXX, df, p4, K, 1:3, shockIdx, labels)];

% --- Display and Save Results ---
ResultsTbl = struct2table(res);
fprintf('\n=== Exercise 4: Granger-causality (VAR(%d)) ===\n', p4);
disp(ResultsTbl);

outdir_ex4 = fileparts(romer_path);
writetable(ResultsTbl, fullfile(outdir,'4_romer_granger_results.csv'));


%% ========================= Exercise 5 =========================

% ---- Settings ----
T5       = 250;
R5       = 5000;
sigma2_5 = 0.6;
alpha0_5 = 0.5;
beta1_5  = 0.05;

% Critical values
z975 = 1.95996; % Two-sided 5% Normal
t975_a = tinv(0.975, T5-1-2);
t975_b = tinv(0.975, T5-1-3);

%%% Question (a): Test H0: rho=1 in a model with an intercept
t_rho_a = zeros(R5,1);
for r = 1:R5
    y = simulate_ar1(T5, 1, sigma2_5, alpha0_5, 0); % RW with drift
    stats = run_ols(y(2:end), y(1:end-1), true);
    t_rho_a(r) = (stats.b(2) - 1) / stats.se(2);
end

rejN_2s_a = mean(abs(t_rho_a) > z975);
rejT_2s_a = mean(abs(t_rho_a) > t975_a);

fprintf('\n=== Exercise 5(a): Intercept only ===\n');
fprintf('mean(t) = %.3f, sd(t) = %.3f, skew = %.3f, kurt = %.3f\n', ...
    mean(t_rho_a), std(t_rho_a), skewness(t_rho_a), kurtosis(t_rho_a));
fprintf('Rej@5%% (two-sided): Normal = %.3f, Student(df=%d) = %.3f\n', ...
    rejN_2s_a, T5-1-2, rejT_2s_a);

plot_t_hist_with_normal(t_rho_a, z975, ...
    '(5a) $t$-stat for $H_0:\ \rho=1$ (levels, intercept only)', ...
    exportFig, '5a_tstat_hist_levels_intercept.pdf');
normal_qqplot_simple(t_rho_a, ...
    '(5a) Normal QQ-plot of $t$-stat ($H_0:\ \rho=1$)', ...
    exportFig, '5a_tstat_normal_qq.pdf');


%%% Question (b): Test H0: rho=1 in a model with intercept and trend
t_rho_b = zeros(R5,1);
tvec5 = (1:T5)';
for r = 1:R5
    y_rw = simulate_ar1(T5, 1, sigma2_5, alpha0_5, 0);
    y = y_rw + beta1_5 * tvec5; % Add deterministic trend
    
    yt = y(2:end);
    X = [tvec5(2:end), y(1:end-1)];
    stats = run_ols(yt, X, true); % OLS on [1, t, y_lag]
    
    t_rho_b(r) = (stats.b(3) - 1) / stats.se(3);
end

rejN_2s_b = mean(abs(t_rho_b) > z975);
rejT_2s_b = mean(abs(t_rho_b) > t975_b);

fprintf('\n=== Exercise 5(b): Intercept + trend ===\n');
fprintf('mean(t) = %.3f, sd(t) = %.3f, skew = %.3f, kurt = %.3f\n', ...
    mean(t_rho_b), std(t_rho_b), skewness(t_rho_b), kurtosis(t_rho_b));
fprintf('Rej@5%% (two-sided): Normal = %.3f, Student(df=%d) = %.3f\n', ...
    rejN_2s_b, T5-1-3, rejT_2s_b);

plot_t_hist_with_normal(t_rho_b, z975, ...
    '(5b) $t$-stat for $H_0:\ \rho=1$ (levels, intercept + trend)', ...
    exportFig, '5b_tstat_hist_levels_trend.pdf');
normal_qqplot_simple(t_rho_b, ...
    '(5b) Normal QQ-plot of $t$-stat ($H_0:\ \rho=1$) with trend', ...
    exportFig, '5b_tstat_normal_qq.pdf');


%% ------------------------ Helper Functions --------------------------

% General Purpose Helpers
function stats = run_ols(y, x, include_const)
% A unified OLS function.
% INPUT:
%   y: dependent variable (T x 1)
%   x: regressors (T x k)
%   include_const: boolean, true to add a constant
% OUTPUT:
%   stats: a struct with b, se, tstat, R2, resid, DW
    if nargin < 3, include_const = true; end
    T = size(y,1);
    X = x;
    if include_const, X = [ones(T,1), x]; end
    
    k = size(X,2);
    try
        XX_inv = (X'*X) \ eye(k);
    catch ME
        if strcmp(ME.identifier, 'MATLAB:nearlySingularMatrix')
            % Handle multicollinearity gracefully
            stats.b = nan(k,1); stats.se = nan(k,1); stats.tstat = nan(k,1);
            stats.R2 = nan; stats.resid = nan(T,1); stats.DW = nan;
            return;
        else
            rethrow(ME);
        end
    end
    
    stats.b = XX_inv * (X'*y);
    stats.resid = y - X * stats.b;
    s2 = (stats.resid' * stats.resid) / (T - k);
    
    stats.V = s2 * XX_inv;
    stats.se = sqrt(diag(stats.V));
    stats.tstat = stats.b ./ stats.se;
    
    y_demeaned = y - mean(y);
    stats.R2 = 1 - (stats.resid' * stats.resid) / (y_demeaned' * y_demeaned);
    
    d_resid = diff(stats.resid);
    stats.DW = (d_resid' * d_resid) / (stats.resid' * stats.resid);
end

function Y = simulate_ar1(T, phi, sigma2, mu, Y0)
% Simulates an AR(1) process: Y_t - c = phi*(Y_{t-1} - c) + eps_t
% where c is the unconditional mean for a stationary process.
% If phi=1, this becomes Y_t = mu + Y_{t-1} + eps_t (RW with drift mu).
    if nargin < 5, Y0 = mu / (1-phi); end % Start at unconditional mean if stable
    if abs(phi) >= 1, Y0 = 0; end % Start at 0 for non-stationary
    
    eps = sqrt(sigma2) * randn(T,1);
    Y = zeros(T,1);
    
    if abs(1-phi) > 1e-8
        c = mu / (1-phi); % Unconditional mean for stationary process
        Y(1) = c + phi*(Y0 - c) + eps(1);
        for t = 2:T
            Y(t) = c + phi*(Y(t-1) - c) + eps(t);
        end
    else % Random walk case (phi=1)
        Y(1) = Y0 + mu + eps(1);
        for t = 2:T
            Y(t) = Y(t-1) + mu + eps(t);
        end
    end
end

% Exercise 2 Helpers
function summary = summarize_case(b1, tstat, R2, DW, tcrit)
    summary.mean_beta1 = mean(b1, 'omitnan');
    summary.sd_beta1 = std(b1, 'omitnan');
    summary.rej_H0_at_5pct = mean(abs(tstat) > tcrit, 'omitnan');
    summary.mean_R2 = mean(R2, 'omitnan');
    summary.mean_DW = mean(DW, 'omitnan');
end

% Exercise 3 Helpers
function [XY, Y_only] = simulate_dgp_invertibility(T, beta, SigU)
% Simulate from the DGP in Exercise 3
    s_eta = sqrt(SigU(1,1)); s_eps = sqrt(SigU(2,2));
    eta = s_eta * randn(T,1); eps = s_eps * randn(T,1);
    x = eta + [0; 0; eps(1:end-2)];
    y = beta/(1-beta) * eta + (beta^2/(1-beta)) * eps + beta * [0; eps(1:end-1)];
    XY = [x y];
    if nargout>1, Y_only = y; end
end

function [Acomp, P] = estimate_var_chol(Y, p)
% Estimate VAR(p) with intercept and return companion matrix and Cholesky of Sigma_u
    [T, n] = size(Y);
    Xlag = mlag(Y, p);
    X = [ones(T,1), Xlag];
    Ytrim = Y(p+1:end, :); Xtrim = X(p+1:end, :);
    
    B = Xtrim \ Ytrim;
    U = Ytrim - Xtrim * B;
    SigmaU = (U' * U) / (size(U,1) - size(B,1));
   
    A = B(2:end,:).'; % A = [A1, A2, ..., Ap]
    Acomp = [A; eye(n*(p-1)), zeros(n*(p-1), n)];
    P = chol(SigmaU, 'lower');
end

function Xlags = mlag(X, p)
% Create lag matrix [L1(X) ... Lp(X)]
    [T, n] = size(X);
    Xlags = zeros(T, n*p);
    for k = 1:p
        Xlags(k+1:end, (n*(k-1)+1):n*k) = X(1:end-k, :);
    end
end

function TH = var_irf_from_companion(Acomp, P, H)
% Compute structural IRFs from companion matrix and Cholesky factor
    np = size(Acomp,1); n = size(P,1);
    J = [eye(n), zeros(n, np-n)];
    TH = zeros(n,n,H+1);
    A_pow = eye(np);
    for h = 0:H
        Phi_h = J * A_pow * J';
        TH(:,:,h+1) = Phi_h * P;
        A_pow = A_pow * Acomp;
    end
end

function trueIRF = compute_true_irfs_ex3(beta, SigU, H)
% Compute the true analytical IRFs for Exercise 3
    trueIRF = zeros(2,2,H+1);
    s_eps = sqrt(SigU(2,2));
    % Shock 1: eta_t (unit variance)
    trueIRF(1,1,1) = 1;
    trueIRF(2,1,1) = beta/(1-beta);
    % Shock 2: eps_t/s_eps (unit variance)
    trueIRF(1,2,3) = s_eps; % Response at h=2
    trueIRF(2,2,1) = (beta^2/(1-beta)) * s_eps; % Response at h=0
    trueIRF(2,2,2) = beta * s_eps; % Response at h=1
end

function plot_irfs_ex3(trueIRF, meanIRF, loIRF, hiIRF, p, Nmc, T, exportFig)
% Plotting function for Exercise 3 results
    h = 0:size(trueIRF,3)-1;
    labelsVar   = {'$x_t$','$y_t$'};
    labelsShock = {'$\eta$ (unit var)','$\varepsilon/\sqrt{0.8}$ (unit var)'};
    fh = figure('Position',[100 100 940 720]);
    tlo = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
    for iv = 1:2
        for js = 1:2
            nexttile; hold on; grid on
            fill([h, fliplr(h)], [squeeze(loIRF(iv,js,:))' fliplr(squeeze(hiIRF(iv,js,:))')], ...
                 [0.8 0.85 1.0], 'EdgeColor','none', 'FaceAlpha',0.6);
            plot(h, squeeze(meanIRF(iv,js,:)), '-', 'LineWidth',1.8);
            plot(h, squeeze(trueIRF(iv,js,:)), '--k', 'LineWidth',1.6);
            xlabel('Horizon $h$'); ylabel('Response');
            title(sprintf('%s to %s', labelsVar{iv}, labelsShock{js}));
            if iv==1 && js==1, legend('95\% band (MC)','MC mean','True','Location','best'); end
        end
    end
    title(tlo, sprintf('Exercise 3: IRFs (VAR(%d), N=%d, T=%d), Cholesky', p, Nmc, T));
    exportFig(fh, sprintf('3_irfs_all_VAR%d_N%d_T%d.pdf', p, Nmc, T));
    close(fh);
end

% Exercise 4 Helpers
function row = run_gc_test(B, s2, invXX, df, p, K, cause_vars, effect_var, labels)
% Helper to run a single Granger-causality Wald test
    n = size(B,2);
    posLag = @(v) 1 + ( (0:p-1)' * n + v );
    
    bj = B(:, effect_var);
    s2j = s2(effect_var);
    
    cols = posLag(cause_vars);
    cols = cols(:); % Ensure it's a column vector
    
    r = numel(cols);
    R_mat = zeros(r, K);
    for i = 1:r, R_mat(i, cols(i)) = 1; end
    
    [W, p_chi2, F, p_F] = waldTest_ols(bj, s2j, invXX, R_mat, df);
    
    cause_label = strjoin(labels(cause_vars), ",");
    if numel(cause_vars)>1, cause_label = "All{" + cause_label + "}"; end
    
    row = struct('Cause', cause_label, 'Arrow', "→", 'Effect', labels(effect_var), ...
                 'NumLags', p, 'NumRestrictions', r, 'Wald_chi2', W, 'p_chi2', p_chi2, ...
                 'Fstat', F, 'p_F', p_F);
end

function [W, pchi2, Fstat, pF] = waldTest_ols(b, s2, invXX, R, df)
% Classic Wald test for H0: R*b=0
    r = size(R,1);
    Rb = R * b;
    V_Rb = R * invXX * R';
    W = Rb' / (s2 * V_Rb) * Rb;
    Fstat = W / r;
    pF = 1 - fcdf(Fstat, r, df);
    pchi2 = 1 - chi2cdf(W, r);
end

% Exercise 5 Helpers
function plot_t_hist_with_normal(tstats, zcrit, ttl, exportFig, outname)
% Plot histogram of t-stats with N(0,1) overlay and critical value lines
    fh = figure('Position', [100 100 860 420]); hold on; grid on
    histogram(tstats, 55, 'Normalization','pdf', 'DisplayName', 'Empirical');
    
    xl = xlim;
    xx = linspace(xl(1), xl(2), 400);
    plot(xx, normpdf(xx, 0, 1), '-', 'LineWidth', 1.6, 'DisplayName', 'Normal(0,1)');
    
    xline([-zcrit, zcrit], '--', {'$-z_{0.975}$', '$z_{0.975}$'}, ...
        'LabelVerticalAlignment','bottom', 'HandleVisibility','off');
    
    title(ttl);
    xlabel('$t = (\hat{\rho}-1)/\mathrm{se}(\hat{\rho})$'); ylabel('Density');
    legend('Location','best');
    
    exportFig(fh, outname);
    close(fh);
end

function normal_qqplot_simple(data, ttl, exportFig, outname)
% Toolbox-free Normal QQ-plot
    fh = figure('Position', [100 100 560 420]);
    
    n = numel(data);
    p = ((1:n)' - 0.5) / n; % Theoretical probabilities
    q_normal = sqrt(2) * erfinv(2*p - 1); % Standard normal quantiles
    
    scatter(q_normal, sort(data), 15, 'filled');
    grid on; hold on;
    
    % Add reference line
    q_data = quantile(data, [0.25, 0.75]);
    q_theory = norminv([0.25, 0.75]);
    b = (q_data(2)-q_data(1))/(q_theory(2)-q_theory(1));
    a = q_data(1) - b*q_theory(1);
    ref_x = xlim;
    plot(ref_x, a + b*ref_x, 'r-');
    
    title(ttl);
    xlabel('Standard Normal Quantiles');
    ylabel('Sample Quantiles');
    
    exportFig(fh, outname);
    close(fh);
end