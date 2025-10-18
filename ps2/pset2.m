%% 20532 Macroeconometrics | Problem Set 2
%
% ---------------------------------------------------------------
% Author: Stefano Graziosi
% Date: 2025-09-22
% ---------------------------------------------------------------

%% Housekeeping & graphics style 
clear; clc; close all; format compact
outdir = fullfile(pwd,'ps2/output');                                        % Output folder -> Update for each problem set
if ~exist(outdir,'dir'), mkdir(outdir); end

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
chi2_95  = 3.8414588207;                                % 95% critical for chi-square(1)
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
fh_a = figure('Position', figPos, 'Renderer', 'painters');
histogram(phi_hat_a, histBins, 'Normalization','pdf'); grid on; hold on
xline(1,'--','True $\phi=1$','LabelVerticalAlignment','bottom');
xlabel('$\hat{\phi}$'); ylabel('Density')
title('(a) Empirical distribution of OLS $\hat{\phi}$ under unit root ($T=250$)')
drawnow;
exportFig(fh_a,'1a_phi_hat_hist.png');
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
fh_b = figure('Position', figPos, 'Renderer', 'painters');
histogram(t_b, histBins, 'Normalization','pdf'); grid on; hold on
xline(z_5_left, '--', 'Normal 5% one-sided crit','LabelVerticalAlignment','bottom');
xlabel('$t(\hat{\rho})$'); ylabel('Density');
title('(b) DF $t$-stat under unit root, intercept included')
drawnow;
exportFig(fh_b,'1b_t_hist.png');
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
rej_diff = nan;
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
writetable(Summary, fullfile(outdir,'spurious_regression_summary.csv'));
disp(Summary);

%% ----------- Plots: t-statistics & R^2 distributions -----------
% t-stat histograms
for k = 1:K
    fh = figure('Position', figPos, 'Renderer', 'painters'); grid on; hold on
    histogram(t_all{k}, histBins, 'Normalization','pdf'); 
    xline(-tcrit, '--', '$-t_{0.975}$','LabelVerticalAlignment','bottom'); xline(tcrit, '--', '$t_{0.975}$','LabelVerticalAlignment','bottom');
    title(sprintf('t-stat of slope: %s (R=%d, T=%d)', caseNames(k), R, T));
    xlabel('$t(\hat{a}_1)$'); ylabel('Density');
    drawnow;
    exportFig(fh, sprintf('tstat_hist_case%d.png', k));
    close(fh);
end

% R^2 histograms
for k = 1:K
    fh = figure('Position', figPos, 'Renderer', 'painters'); grid on; hold on
    histogram(R2_all{k}, histBins, 'Normalization','pdf'); 
    title(sprintf('$R^2$ distribution: %s (R=%d, T=%d)', caseNames(k), R, T));
    xlabel('$R^2$'); ylabel('Density');
    drawnow;
    exportFig(fh, sprintf('R2_hist_case%d.png', k));
    close(fh);
end

% Quick illustration for one replication of Case 3 (time series + scatter)
y = cumsum(randn(T,1)); z = cumsum(randn(T,1));
[b, ~, ~, ~, u] = ols_with_const(y, z);
fh = figure('Position',[100 100 880 380], 'Renderer', 'painters');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile; plot([y z]); grid on; legend('$y_t$','$z_t$','Location','best');
title('Case 3 example: two independent random walks'); xlabel('t');
nexttile; scatter(z,y,15,'filled'); grid on; hold on
xline(mean(z),'--'); yline(mean(y),'--');
title(sprintf('Scatter (\\beta_1=%.2f), residuals are I(1)', b(2)));
xlabel('$z_t$'); ylabel('$y_t$');
drawnow;
exportFig(fh,'case3_example_scatter_timeseries.png');
close(fh);

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

