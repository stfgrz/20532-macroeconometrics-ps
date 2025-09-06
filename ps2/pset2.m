%% Setup (applies to the whole file)
clear; clc; close all;
rng(42); % reproducibility

%% EXERCISE 1 - A)
phi = 1;
T = 250;
n_sim = 10000;
sigma_eps = 1;

% Random walk can be generated via cumulative sums
E = sigma_eps * randn(T, n_sim);
Yfull = cumsum(E, 1);            % y(1)=E(1), start at 0 is immaterial asymptotically
Ylag  = Yfull(1:end-1, :);
Ynow  = Yfull(2:end,   :);

% OLS slope without intercept (beta = sum(xy)/sum(x^2))
num  = sum(Ylag .* Ynow, 1);
den  = sum(Ylag .^ 2,    1);
phihat = (num ./ den).';

figure; histogram(phihat, 50);
title('Empirical Distribution of OLS Estimator \phi (RW, no intercept)');
xlabel('\phi'); ylabel('Frequency');

%% EXERCISE 1 - B)
% DF regression with intercept: Δy_t = α + ρ y_{t-1} + u_t
rej_count_normal = 0;
alpha_sig = 0.05;
z_alpha   = norminv(alpha_sig);  % left tail (≈ -1.645)
t_stats   = zeros(n_sim, 1);

for idx = 1:n_sim
    epsi = sigma_eps * randn(T, 1);
    y    = cumsum(epsi);          % phi = 1 random walk, y(1)=epsi(1)
    dy   = diff(y);
    ylag = y(1:end-1);

    X = [ones(T-1,1), ylag];
    [b, se, ~, ~, s2, XtX_inv] = ols(dy, X);   % helper below
    rhohat = b(2);
    se_rho = sqrt(s2 * XtX_inv(2,2));
    t_stat = rhohat / se_rho;                  % DF t-stat (intercept only)
    t_stats(idx) = t_stat;

    % "textbook" (incorrect for DF) normal rejection, kept for comparison
    if t_stat < z_alpha
        rej_count_normal = rej_count_normal + 1;
    end
end

rej_rate_normal = rej_count_normal / n_sim;
fprintf('1B: Normal left-tail rejection at 5%% (incorrect benchmark): %.2f%%%%\n', 100*rej_rate_normal);

figure; histogram(t_stats, 50);
title('Distribution of DF t-statistic (model: intercept)');
xlabel('t-statistic'); ylabel('Frequency');

%% EXERCISE 1 - C)
pct = [1 5 10];
percentiles = prctile(t_stats, pct);
disp('Percentiles of the empirical DF t-stat (intercept):');
disp(table(pct.', percentiles.', 'VariableNames', {'Percentile','t_Statistic'}));

%% EXERCISE 1 - D)
% Random walk with drift: y_t = μ + y_{t-1} + ε_t
mu  = 1;
phihat_drift = zeros(n_sim, 1);
rej_count_df = 0;

for sim = 1:n_sim
    e = sigma_eps * randn(T,1);
    y = zeros(T,1);
    for t = 2:T
        y(t) = mu + y(t-1) + e(t);
    end

    % OLS slope of y_t on y_{t-1} (no intercept), like in 1A
    Ynow = y(2:T);
    Ylag = y(1:T-1);
    phihat_drift(sim) = (Ylag\Ynow);

    % Proper ADF with drift (intercept)
    [h, pValue] = adftest(y, 'model', 'ARD');
    if h == 1 && pValue < 0.05
        rej_count_df = rej_count_df + 1;
    end
end

figure; histogram(phihat_drift, 30);
title('Empirical Distribution of \phî with drift (RW with drift, no intercept in OLS)');
xlabel('\phî'); ylabel('Frequency');

rej_rate_df = rej_count_df / n_sim;
fprintf('1D: ADF(ARD) rejection rate at 5%%: %.2f%%%%\n', 100*rej_rate_df);

%% Exercise 1-E
% F-test for adding y_{t-1} in Δy regression with intercept
rng(7); % new seed to match the spirit of your section
mu = 1; phi = 1; sigma_eps = 1; T = 250; n_sim = 10000;

F_stats = zeros(n_sim,1);
rej_count_F = 0;

for sim = 1:n_sim
    e = sigma_eps * randn(T,1);
    y = zeros(T,1);
    for t = 2:T
        y(t) = mu + phi*y(t-1) + e(t); % RW with drift
    end
    dy = diff(y);
    x_full = [ones(T-1,1), y(1:T-1)];
    [~, ~, r_full, SSR_full] = ols(dy, x_full);

    x_res  = ones(T-1,1);                 % restricted: intercept only
    [~, ~, r_res,  SSR_res ] = ols(dy, x_res);

    q  = 1;                 % one restriction (coefficient on y_{t-1})
    n  = T-1; k = 2;        % T-1 obs, 2 params in the unrestricted model
    F  = ((SSR_res - SSR_full)/q) / (SSR_full/(n-k));
    F_stats(sim) = F;

    if F > finv(0.95, q, n-k)
        rej_count_F = rej_count_F + 1;
    end
end

rejection_rate_F = rej_count_F / n_sim;
critF = finv(0.95, 1, (T-1)-2);

figure;
histogram(F_stats, 50, 'Normalization', 'probability');
hold on; xline(critF, 'r--', 'LineWidth', 1.5);
title('F-statistics: Δy_t on [1, y_{t-1}] vs [1]');
xlabel('F'); ylabel('Proportion'); grid on; hold off;

fprintf('1E: F-test rejection rate at 5%% (F(1,%d) crit=%.3f): %.2f%%%%\n', (T-1)-2, critF, 100*rejection_rate_F);

%% Exercise 1-f
% DF with deterministic trend: simulate null y_t = α + β t + y_{t-1} + ε_t
rng(123);
T = 100;
numSimulations = 10000;
alpha0 = 1; beta0 = 0.5;
tStats = zeros(numSimulations,1);

burn = 50;
for sim = 1:numSimulations
    Tlong = T + burn;
    tvec  = (1:Tlong).';
    epsi  = randn(Tlong,1);
    y     = zeros(Tlong,1);
    for t = 2:Tlong
        y(t) = alpha0 + beta0*t + y(t-1) + epsi(t);   % unit root with trend
    end
    y = y(burn+1:end);           % drop burn-in to wash out y(0)

    dy   = diff(y);
    ylag = y(1:end-1);
    trend= (1:T-1).';

    X = [ones(T-1,1), trend, ylag];
    [b, ~, ~, ~, s2, XtX_inv] = ols(dy, X);
    tStats(sim) = b(3) / sqrt(s2 * XtX_inv(3,3));     % DF t-stat for γ=0 on y_{t-1}
end

critVal5 = prctile(tStats, 5);
critVal1 = prctile(tStats, 1);
fprintf('1f: DF(trend) criticals (empirical): 5%%=%.3f, 1%%=%.3f\n', critVal5, critVal1);

% Now test a new trend series
alpha_new = 1.5; beta_new = 0.3;
eps_new   = randn(T,1);
t_values  = (1:T).';
y_new     = zeros(T,1);
for t = 2:T
    y_new(t) = alpha_new + beta_new*t + y_new(t-1) + eps_new(t);
end

dy_new   = diff(y_new);
ylag_new = y_new(1:end-1);
trend_new= (1:T-1).';
X_new    = [ones(T-1,1), trend_new, ylag_new];
[b_new, ~, ~, ~, s2_new, XtX_inv_new] = ols(dy_new, X_new);
t_stat_new = b_new(3) / sqrt(s2_new * XtX_inv_new(3,3));
reject_5 = (t_stat_new < critVal5);
reject_1 = (t_stat_new < critVal1);

fprintf('1f: DF(trend) t-stat on new series: %.3f | reject@5%%=%d, reject@1%%=%d\n', t_stat_new, reject_5, reject_1);

figure;
histogram(tStats, 'Normalization', 'pdf'); hold on;
[f, xi] = ksdensity(tStats); plot(xi, f, 'LineWidth', 2);
xline(critVal5, 'r--', 'LineWidth', 2, 'DisplayName', '5%');
xline(critVal1, 'g--', 'LineWidth', 2, 'DisplayName', '1%');
title('DF(trend) empirical distribution (null)'); xlabel('t'); ylabel('Density');
legend('Histogram','Kernel Density','5%','1%'); grid on; hold off;

figure;
plot(t_values, y_new, 'LineWidth', 1.5); hold on;
plot(t_values, alpha_new + beta_new*t_values, 'r--', 'LineWidth', 1.2);
title('New deterministic trend with unit root'); xlabel('Time'); ylabel('y_t');
legend('y_t','trend','Location','best'); grid on;

%% Exercise 2
rng(42);
T = 250;
n_sim = 10000;
sigma_eps = 1;

phi_Y = [0.1, 1, 1, 1];
phi_Z = [0.1, 0.1, 1, 1];

coef = zeros(n_sim, 4);
t_stat_reg = zeros(n_sim, 4);
R2 = zeros(n_sim, 4);

for sim = 1:n_sim
    for casus = 1:4
        Y = zeros(T,1);  eY = sigma_eps * randn(T,1);
        for t = 2:T, Y(t) = phi_Y(casus)*Y(t-1) + eY(t); end

        if casus == 4
            eZ = sigma_eps * randn(T,1);
            Z  = Y + eZ;
        else
            Z = zeros(T,1); eZ = sigma_eps * randn(T,1);
            for t = 2:T, Z(t) = phi_Z(casus)*Z(t-1) + eZ(t); end
        end

        [beta, tstat, r2] = run_regression(Y, Z);   % updated helper below
        coef(sim, casus)     = beta;
        t_stat_reg(sim,casus)= tstat;
        R2(sim, casus)       = r2;
    end
end

case_labels = {'Case 1','Case 2','Case 3','Case 4'}';
summary_table = table(case_labels, mean(coef).', mean(t_stat_reg).', mean(R2).', ...
    'VariableNames', {'Case','Mean_Coefficient','Mean_t_Statistic','Mean_R2'});
disp(summary_table);

%% Exercise 3: Granger causality tests
% NOTE: set your own path below
file_path = 'C:\Users\allic\Downloads\Romer_Romer.xlsx'; % <--- change to your path
data = readtable(file_path, 'VariableNamingRule','preserve');
data.Properties.VariableNames = {'Time','Inflation','Unemployment','FFR','Romer_Shocks'};

Y = [data.Inflation, data.Unemployment, data.FFR, data.Romer_Shocks];
numseries = size(Y,2);
numLags = 4;

Y0  = Y(1:numLags, :);
Yest= Y(numLags+1:end, :);

VARmodel = varm(numseries, numLags);
VARmodel.SeriesNames = {'Inflation','Unemployment','FFR','Romer_Shocks'};
BestMdl = estimate(VARmodel, Yest, 'Y0', Y0);

fprintf('\nGranger causality test results with four lags:\n');

[h, pTbl] = gctest(BestMdl, 'Cause', 4, 'Effect', 1);
p = pTbl.PValue;
fprintf('Romer shocks -> Inflation, p=%.4f | %s\n', p, ternary(h,'Reject','Fail to reject'));

[h, pTbl] = gctest(BestMdl, 'Cause', 4, 'Effect', 2);
p = pTbl.PValue;
fprintf('Romer shocks -> Unemployment, p=%.4f | %s\n', p, ternary(h,'Reject','Fail to reject'));

[h, pTbl] = gctest(BestMdl, 'Cause', 4, 'Effect', 3);
p = pTbl.PValue;
fprintf('Romer shocks -> FFR, p=%.4f | %s\n\n', p, ternary(h,'Reject','Fail to reject'));

for j = 1:3
    [h, pTbl] = gctest(BestMdl, 'Cause', j, 'Effect', 4);
    p = pTbl.PValue;
    fprintf('%s -> Romer shocks, p=%.4f | %s\n', BestMdl.SeriesNames{j}, p, ternary(h,'Reject','Fail to reject'));
end

%% Exercise 4
beta = 0.6;
T = 500;
N = 1000;
num_periods = 10;

% Structural shocks covariance
cov_u = [1, 0; 0, 0.8];

% Generate one realization (for the "true" IRFs & a baseline VAR)
rng(100);
shocks = mvnrnd([0, 0], cov_u, T).';  % 2×T

x = zeros(1,T); y = zeros(1,T);
for t = 3:T
    x(t) = shocks(1,t) + shocks(2,t-2);
    y(t) = (beta/(1-beta))*shocks(1,t) + (beta^2/(1-beta))*shocks(2,t) + beta*shocks(2,t-1);
end
data = [x' y'];

% True IRFs
true_irf_eta      = zeros(2, num_periods);
true_irf_epsilon  = zeros(2, num_periods);
for h = 1:num_periods
    true_irf_eta(1,h)     = (h == 1);                   % x ← η at h=0
    true_irf_eta(2,h)     = (beta/(1-beta)) * (h == 1); % y ← η at h=0

    true_irf_epsilon(1,h) = (h == 3) * 1;               % x ← ε at h=2 (coefficient 1)
    if     h == 1, true_irf_epsilon(2,h) =  beta^2/(1-beta);
    elseif h == 2, true_irf_epsilon(2,h) =  beta;
    else,           true_irf_epsilon(2,h) =  0;
    end
end

% Estimate baseline VAR(4) and compute residuals
lags = 4;
model = varm(2, lags);
EstMdl = estimate(model, data);
% irf returns an H×K×K array of responses to one-s.d. reduced-form shocks
% If you prefer Cholesky-orthogonalized IRFs (identification), use:
% irf(EstMdl, 'NumObs', num_periods, 'Method', 'orthogonalized');
IRF0 = irf(EstMdl, 'NumObs', num_periods);

% Monte Carlo over the DGP + re-estimated VARs and their IRFs
irf_estimates = zeros(num_periods, 2, 2, N);
parfor (iMC = 1:N)
    shocks_i = mvnrnd([0, 0], cov_u, T).';
    xi = zeros(1,T); yi = zeros(1,T);
    for t = 3:T
        xi(t) = shocks_i(1,t) + shocks_i(2,t-2);
        yi(t) = (beta/(1-beta))*shocks_i(1,t) + (beta^2/(1-beta))*shocks_i(2,t) + beta*shocks_i(2,t-1);
    end
    dat_i   = [xi' yi'];
    mdl_i   = estimate(varm(2, lags), dat_i);
    irf_i   = irf(mdl_i, 'NumObs', num_periods);
    irf_estimates(:,:,:,iMC) = irf_i;
end

irf_mean = mean(irf_estimates, 4);
irf_5th  = prctile(irf_estimates, 5,  4);
irf_95th = prctile(irf_estimates, 95, 4);

% Plot true vs estimated IRFs
figure;
for iVar = 1:2
    for jShock = 1:2
        subplot(2,2,(iVar-1)*2 + jShock);
        if jShock == 1
            plot(0:num_periods-1, true_irf_eta(iVar,:), 'k', 'LineWidth', 2); hold on;
        else
            plot(0:num_periods-1, true_irf_epsilon(iVar,:), 'k', 'LineWidth', 2); hold on;
        end
        plot(0:num_periods-1, squeeze(irf_mean(:,iVar,jShock)), 'b', 'LineWidth', 1.5);
        plot(0:num_periods-1, squeeze(irf_5th(:,iVar,jShock)), 'b--');
        plot(0:num_periods-1, squeeze(irf_95th(:,iVar,jShock)), 'b--');
        title(sprintf('Shock %d \x2192 var %d', jShock, iVar));
        xlabel('Horizon'); ylabel('Response'); grid on; hold off;
        legend('True','Mean','5th','95th','Location','best');
    end
end

% Invertibility check
beta = 0.6;
ma_poly_x = [1 0 1];                 % 1 + L^2
% For y: Θ(L) ∝ 1 + ((1-β)/β) L   (scale-free for roots)
ma_poly_y = [1 (1-beta)/beta];

roots_x = roots(ma_poly_x);
roots_y = roots(ma_poly_y);
disp('Roots of MA polynomial for x_t (1 + L^2):'); disp(roots_x);
disp('Roots of MA polynomial for y_t (normalized 1 + cL):'); disp(roots_y);

invertible_x = all(abs(roots_x) > 1);
invertible_y = all(abs(roots_y) > 1);

disp( ternary(invertible_x,'x_t MA is invertible.','x_t MA is NOT invertible.') );
disp( ternary(invertible_y,'y_t MA is invertible.','y_t MA is NOT invertible.') );

%% Functions used
function [beta, t_stat, R2] = run_regression(Y, Z)
    % OLS of Y on [1 Z], with robust fundamentals
    X = [ones(length(Z),1), Z];
    [b, ~, ~, ~, s2, XtX_inv, resid] = ols(Y, X);
    beta = b(2);
    se   = sqrt(s2 * XtX_inv(2,2));
    t_stat = beta / se;

    SS_res = sum(resid.^2);
    SS_tot = sum((Y - mean(Y)).^2);
    R2 = 1 - SS_res/SS_tot;
end

function [b, se, yhat, e, s2, XtX_inv, resid] = ols(y, X)
    % Basic OLS with classic covariance
    XtX = X' * X;
    XtX_inv = inv(XtX);
    b = XtX_inv * (X' * y);
    yhat = X * b;
    resid = y - yhat;
    n = size(X,1); k = size(X,2);
    s2 = (resid' * resid) / (n - k);
    se = sqrt(diag(s2 * XtX_inv));
    e  = resid; % alias
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
