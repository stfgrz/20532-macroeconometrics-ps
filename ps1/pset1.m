%% PS1 — 20532 Macroeconometrics 
% Problem Set 1
% ---------------------------------------------------------------
% Author: Stefano Graziosi
% Date: 2025-09-22
% ---------------------------------------------------------------

%% Housekeeping & graphics style 
clear; clc; close all; format compact
outdir = fullfile(pwd,'ps1/output');   % Output folder
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

%% ===================== Exercise 1 =====================
% Simulate AR(1) by loop and by filter, and compare
% Y_t = mu + phi*(Y_{t-1} - mu) + eps_t,  eps_t ~ N(0, sigma2)

rng(20532,'twister');       % Reproducibility for Ex.1
T      = 500;
phi1   = 0.4;
sigma2_1 = 0.2;
mu1    = 0;                 % E[Y_t] = 0
Y0     = 0;                 % matching starting condition for both methods

% Use the SAME innovation sequence for both methods
eps1 = sqrt(sigma2_1) * randn(T,1);

% (a) For-loop simulation
Y_loop = simulate_ar1_loop(T, phi1, sigma2_1, mu1, Y0, eps1);

% (b) Using `filter` (exact handling of initial condition)
Y_filt = simulate_ar1_filter(T, phi1, sigma2_1, mu1, Y0, eps1);

% (c) Check equality (up to machine precision)
diff_vec = Y_loop - Y_filt;
max_abs_diff = max(abs(diff_vec));
fprintf('[Ex.1] Max |difference| between methods: %.3e\n', max_abs_diff);

% Plot: overlay the two series
fh1 = figure('Position',[100 100 800 400]);
plot(1:T, Y_loop, '-', 'DisplayName','Loop'); hold on
plot(1:T, Y_filt, '--', 'DisplayName','Filter'); grid on
xlabel('t'); ylabel('$Y_t$')
title('Exercise 1: AR(1) via Loop vs. Filter (Overlay)')
legend('Location','best')
exportFig(fh1,'ex1_overlay.png');

% Plot: difference
fh2 = figure('Position',[100 100 800 350]);
plot(1:T, diff_vec, '-'); grid on
xlabel('t'); ylabel('$Y^{loop}_t - Y^{filter}_t$')
title(sprintf('Exercise 1: Difference, Max = %.1e', max_abs_diff))
exportFig(fh2,'ex1_difference.png');


%% ===================== Exercise 2 =====================
% AR(1) with phi=0.6, sigma^2=0.4, E[Y_t]=3, start Y0=20

rng(23456,'twister');              % Reproducibility for Ex.2
T      = 500;
phi2   = 0.6;
sigma2_2 = 0.4;
mu2    = 3;        % unconditional mean
Y0_far = 20;       % starting far from mean

% Simulate with a for-loop (explicit control over initial condition)
eps2 = sqrt(sigma2_2) * randn(T,1);
Y2   = simulate_ar1_loop(T, phi2, sigma2_2, mu2, Y0_far, eps2);

% Plot the sample path and the unconditional mean
fh3 = figure('Position',[100 100 900 360]);
plot(1:T, Y2, '-', 'DisplayName','$Y_t$'); hold on; grid on
yline(mu2, '--', '$\mathrm{E}[Y_t]=\mu=3$', 'Interpreter','latex', 'LabelVerticalAlignment','bottom', 'DisplayName', '$\mathrm{E}[Y_t]=\mu=3$')
xlabel('t'); ylabel('$Y_t$')
title('Exercise 2: AR(1) Path with Initial Condition Far from Mean')
legend('Location','best')
exportFig(fh3,'ex2_path_far_from_mean.png');

% "Proper" stationary realization: use burn-in and then drop it
B   = 500;                              % burn-in length
TT  = T + B;
eps2b = sqrt(sigma2_2) * randn(TT,1);
Y2b    = simulate_ar1_loop(TT, phi2, sigma2_2, mu2, Y0_far, eps2b);
Y2_stat = Y2b(B+1:end);                 % drop initial transient

% Plot the post-burn-in sample path and the mean
fh4 = figure('Position',[100 100 900 360]);
plot(1:T, Y2_stat, '-', 'DisplayName','$Y_t$ after burn-in'); hold on; grid on
yline(mu2, '--', '$\mathrm{E}[Y_t]=\mu=3$', 'Interpreter','latex', 'LabelVerticalAlignment','bottom', 'DisplayName', '$\mathrm{E}[Y_t]=\mu=3$')
xlabel('t'); ylabel('$Y_t$')
title(sprintf('Exercise 2: Stationary Sample Path After Burn-in (B = %d)',B))
legend('Location','best')
exportFig(fh4,'ex2_path_after_burnin.png');


%% ===================== Exercise 3 =====================
% Empirical distribution of OLS estimator; t-test of H0: phi=0
% DGP: AR(1) with phi=0.4, T=250.

rng(34567,'twister');              % Reproducibility for Ex.3
T      = 250;
phi3   = 0.4;
sigma2_3 = 1.0;      % explicit
mu3    = 0;
R      = 5000;       % number of Monte Carlo replications
B      = 300;        % short burn-in for stationarity

phi_hat = zeros(R,1);
tstat   = zeros(R,1);

for r = 1:R
    % Innovations and simulation length with burn-in
    TT   = T + B;
    eps3 = sqrt(sigma2_3) * randn(TT,1);

    % Start at the mean (mu3) + burn-in
    Ytmp = simulate_ar1_loop(TT, phi3, sigma2_3, mu3, mu3, eps3);
    Y    = Ytmp(B+1:end);                 % keep last T observations

    % OLS in Y_t = phi * Y_{t-1} + u_t  (no intercept; mean is zero)
    ylag = Y(1:end-1);
    yt   = Y(2:end);
    X    = ylag;                           % (T-1) x 1
    bhat = (X' * X) \ (X' * yt);
    uhat = yt - X * bhat;

    % --- Correct degrees of freedom: nu = (T-1) - 1 = T - 2 ---
    nu   = (T - 1) - 1;
    s2   = (uhat' * uhat) / nu;            % unbiased sigma_u^2
    se   = sqrt( s2 / (X' * X) );          % std error of bhat

    phi_hat(r) = bhat;
    tstat(r)   = bhat / se;                % test H0: phi = 0
end

% Rejection frequency at 5%
if exist('tinv','file')
    tcrit = tinv(0.975, nu);
else
    tcrit = 1.96;                          % normal approx
end
reject = mean(abs(tstat) > tcrit);

% Report
fprintf('[Ex.3] Monte Carlo with R=%d, T=%d: mean(phi_hat)=%.4f, sd(phi_hat)=%.4f, reject H0 at 5%% = %.3f\n', ...
    R, T, mean(phi_hat), std(phi_hat), reject);

% Save a small summary table
MC_tbl = table(mean(phi_hat), std(phi_hat), reject, 'VariableNames', ...
    {'mean_phi_hat','sd_phi_hat','reject_H0_rate'});
writetable(MC_tbl, fullfile(outdir,'ex3_summary.csv')); % CSV

% Histogram of phi_hat with reference lines
fh5 = figure('Position',[100 100 800 420]);
histogram(phi_hat, 40, 'Normalization','pdf'); hold on; grid on
xline(phi3, '--', 'True $\phi=0.4$', 'LabelVerticalAlignment','bottom', 'Interpreter','latex');
xline(mean(phi_hat), '-', '$\mathrm{mean}(\hat{\phi})$', 'LabelVerticalAlignment','bottom', 'Interpreter','latex');
xlabel('$\hat{\phi}$'); ylabel('Density')
title(['Exercise 3: Empirical Distribution of OLS ', '$\hat{\phi}$', ' (R=', num2str(R), ', T=', num2str(T), ')'], 'Interpreter','latex')
exportFig(fh5,'ex3_phi_hat_hist.png');

%% ===================== Exercise 4 =====================
% Empirical distribution of OLS AR(1) with phi = 0.9 over varying T

rng(45678,'twister');                % Reproducibility for Ex.4
phi      = 0.9;
sigma2_4 = 1.0;                      % Var(eps_t)
mu       = 0;
Ts       = [50, 100, 200, 1000];
R        = 1000;
B        = 500;                      % burn-in to reduce dependence on start

E4_summary = table('Size',[numel(Ts) 5], ...
    'VariableTypes',{'double','double','double','double','double'}, ...
    'VariableNames',{'T','mean_phi_hat','sd_phi_hat','bias','rej_H0_phi0_rate'});

for iT = 1:numel(Ts)
    T = Ts(iT);
    nu = T - 2;                      % df for regression with (T-1) rows, 1 slope

    phi_hat = zeros(R,1);
    tstat   = zeros(R,1);

    for r = 1:R
        % --- simulate AR(1) with burn-in
        TT   = T + B;
        eps  = sqrt(sigma2_4) * randn(TT,1);
        Yall = simulate_ar1_loop(TT, phi, sigma2_4, mu, mu, eps);
        Y    = Yall(B+1:end);        % keep last T observations

        % --- OLS: Y_t = phi * Y_{t-1} + u_t (no intercept since mu=0)
        ylag = Y(1:end-1); 
        yt   = Y(2:end);
        X    = ylag;                  % (T-1)-by-1 regressor
        XX   = X' * X;

        bhat = XX \ (X' * yt);
        uhat = yt - X * bhat;

        % --- correct finite-sample variance and t-stat
        s2 = (uhat' * uhat) / nu;     % unbiased residual variance: RSS/(T-2)
        se = sqrt( s2 / XX );         % std error of slope
        tstat(r)   = bhat / se;       % test H0: phi = 0
        phi_hat(r) = bhat;
    end

    % --- Monte Carlo summaries for this T
    m    = mean(phi_hat);
    sd   = std(phi_hat);
    bias = m - phi;

    if exist('tinv','file')
        tcrit = tinv(0.975, nu);      % two-sided 5% test against H0: phi = 0
    else
        tcrit = 1.96;                 % normal approx
    end
    rej = mean(abs(tstat) > tcrit);

    E4_summary{iT,:} = [T, m, sd, bias, rej];

    % --- Histogram for this T
    fh = figure('Position',[100 100 840 420]);
    histogram(phi_hat, 40, 'Normalization','pdf'); hold on; grid on
    xline(phi, '--', 'True $\phi=0.9$', 'LabelVerticalAlignment','bottom', 'Interpreter','latex');
    xline(m,  '-',  '$\mathrm{mean}(\hat{\phi})$', 'LabelVerticalAlignment','bottom', 'Interpreter','latex');
    xlabel('$\hat{\phi}$', 'Interpreter','latex'); 
    ylabel('Density', 'Interpreter','latex');
    title(sprintf('Exercise 4: OLS on AR(1), $\\phi=0.9$ (T=%d, R=%d)', T, R), 'Interpreter','latex');
    exportFig(fh, sprintf('ex4_hist_T%d.png', T));
end

% Save table
writetable(E4_summary, fullfile(outdir,'ex4_summary.csv'));

%% ===================== Exercise 5 =====================
% OLS of x_t on x_{t-1} when x_t is MA(1) with theta=0.6
% DGP: x_t = eps_t + theta * eps_{t-1},  eps_t ~ N(0, sigma^2)

rng(56789,'twister');              % Reproducibility for Ex.5
theta = 0.6;
sigma2_5 = 1.0;        % explicit
Ts   = [50, 100, 200, 1000];
R    = 1000;
B    = 500;            % burn-in for MA(1)

% Theoretical plim of OLS when regressing x_t on x_{t-1}: rho(1) = theta/(1+theta^2)
plim_a = theta / (1 + theta^2);

E5_summary = table('Size',[numel(Ts) 5], ...
    'VariableTypes',{'double','double','double','double','double'}, ...
    'VariableNames',{'T','mean_a_hat','sd_a_hat','bias_from_plim','theoretical_plim'});

for iT = 1:numel(Ts)
    T = Ts(iT);
    a_hat = zeros(R,1);

    for r = 1:R
        TT   = T + B;
        % Simulate MA(1) with burn-in
        [x_all, ~] = simulate_ma1(TT, theta, sigma2_5, 0);
        x = x_all(B+1:end);

        Xlag = x(1:end-1); xt = x(2:end);
        bhat = (Xlag' * Xlag) \ (Xlag' * xt);  % no constant
        a_hat(r) = bhat;
    end

    m  = mean(a_hat);
    sd = std(a_hat);
    bias = m - plim_a;
    E5_summary{iT,:} = [T, m, sd, bias, plim_a];

    % Plot histogram for this T
    fh = figure('Position',[100 100 840 420]);
    histogram(a_hat, 40, 'Normalization','pdf'); hold on; grid on
    xline(plim_a, '--', sprintf('plim = %.3f', plim_a), 'LabelVerticalAlignment','bottom');
    xline(m,      '-',  '$\mathrm{mean}(\hat{a})$', 'LabelVerticalAlignment','bottom');
    xlabel('$\hat{a}$'); ylabel('Density')
    title(['Exercise 5: OLS on MA(1) Data, ', '$\theta=0.6$', ' (T=', num2str(T), ', R=', num2str(R), ')'], 'Interpreter','latex')
    exportFig(fh, sprintf('ex5_hist_T%d.png', T));
end

% Save table
writetable(E5_summary, fullfile(outdir,'ex5_summary.csv'));

% Optional: plot mean(\hat{a}) across T for a compact summary figure
fh = figure('Position',[100 100 620 360]); grid on; hold on
plot(E5_summary.T, E5_summary.mean_a_hat, '-o', 'DisplayName','$\mathrm{mean}(\hat{a})$');
yline(plim_a, '--', 'DisplayName','theoretical plim');
xlabel('$T$'); ylabel('Mean of $\hat{a}$ across replications')
title('Exercise 5: Convergence of $\mathrm{mean}(\hat{a})$ with $T$', 'Interpreter','latex')
legend('Location','best')
exportFig(fh,'ex5_mean_vs_T.png');


%% ===================== Exercise 6 =====================
% Function to generate ARMA(p,q) via for-loop (with examples)
% The function `arma_sim` below creates T observations from
% an ARMA(p,q) process using explicit recursion.
% -------------------------------------------------------------------
% USAGE EXAMPLES (uncomment to try):
%
% % Example A: ARMA(2,1) with coefficients (stationary & invertible)
% rng(67890,'twister'); T = 400; B = 800; sigma2 = 1; mu = 0;
% ar = [0.5, -0.3];  % phi_1, phi_2
% ma = [0.4];        % theta_1
% [yA, eA] = arma_sim(T + B, sigma2, ar, ma, 'ParamType','coeffs','BurnIn',0,'Mu',mu);
% yA = yA(B+1:end);
% fhA = figure('Position',[100 100 900 320]);
% plot(yA,'-'); grid on; xlabel('t'); ylabel('$y_t$')
% title('Exercise 6: Example ARMA(2,1) sample path')
% exportFig(fhA,'ex6_demo_arma_series.png');
%
% % Example B: ARMA(1,1) with LAG ROOTS (lambda) instead of coefficients
% % AR lag polynomial A(L) = (1 - lambda_AR L), MA lag polynomial B(L) = (1 + lambda_MA L)
% % Then the implied coefficients satisfy A(L) = 1 - phi_1 L, B(L) = 1 + theta_1 L.
% rng(78901,'twister'); T = 400; B = 800; sigma2 = 1; mu = 0;
% lambda_ar = [0.6];    % |lambda|<1 ensures stationarity in this parameterization
% lambda_ma = [0.5];    % |lambda|<1 ensures invertibility for B(L) = prod(1 + lambda L)
% [yB, eB] = arma_sim(T + B, sigma2, lambda_ar, lambda_ma, 'ParamType','roots','BurnIn',0,'Mu',mu);
% yB = yB(B+1:end);
% fhB = figure('Position',[100 100 900 320]);
% plot(yB,'-'); grid on; xlabel('t'); ylabel('$y_t$')
% title('Exercise 6: Example ARMA(1,1) from roots (lag-root parameterization)')
% exportFig(fhB,'ex6_demo_arma_series_roots.png');


%% ------------------------- Helper functions -------------------------
function Y = simulate_ar1_loop(T, phi, sigma2, mu, Y0, eps)
%SIMULATE_AR1_LOOP Simulate AR(1) using explicit recursion (for-loop).
%   Y_t = mu + phi*(Y_{t-1} - mu) + eps_t
%   Inputs:
%       T, phi, sigma2, mu, Y0  — scalars
%       eps  — T-by-1 vector of innovations (optional)
%   Output:
%       Y   — T-by-1 simulated series
    if nargin < 6 || isempty(eps)
        eps = sqrt(sigma2) * randn(T,1);
    end
    Y       = zeros(T,1);
    Y(1)    = mu + phi*(Y0 - mu) + eps(1);
    for t = 2:T
        Y(t) = mu + phi*(Y(t-1) - mu) + eps(t);
    end
end

function Y = simulate_ar1_filter(T, phi, sigma2, mu, Y0, eps)
%SIMULATE_AR1_FILTER Simulate AR(1) using FILTER for the centered process.
%   Let X_t = Y_t - mu, then X_t = phi*X_{t-1} + eps_t.
%   We generate X via filter and then shift back by mu.
%   This implementation adds the exact initial-condition term phi^t * X0
%   so the result matches the loop simulation pointwise.
    if nargin < 6 || isempty(eps)
        eps = sqrt(sigma2) * randn(T,1);
    end
    X0 = Y0 - mu;        % initial condition for centered process
    % Zero-initial-condition filtered component
    Z = filter(1, [1, -phi], eps);
    t = (1:T)';
    X = Z + (phi.^t) * X0;   % exact IC adjustment
    Y = X + mu;
end

function [x, eps] = simulate_ma1(T, theta, sigma2, mu)
%SIMULATE_MA1 Simulate MA(1): x_t = mu + eps_t + theta*eps_{t-1}
    if nargin < 4, mu = 0; end
    eps = sqrt(sigma2) * randn(T,1);
    % Vectorized MA(1): set eps_0 = 0 and use lagged innovations
    eps_lag = [0; eps(1:end-1)];
    x = mu + eps + theta * eps_lag;
end

function [y, e] = arma_sim(T, sigma2, ar_in, ma_in, varargin) %#ok<DEFNU>
%ARMA_SIM Generate T observations from an ARMA(p,q) via for-loop.
%   y_t = mu + sum_{i=1}^p phi_i (y_{t-i} - mu) + e_t + sum_{j=1}^q theta_j e_{t-j}
%   with e_t ~ N(0, sigma2).
%
% Inputs (required):
%   T        — number of observations to RETURN (after optional burn-in)
%   sigma2   — variance of white noise e_t
%   ar_in    — AR parameters: either coefficients [phi_1..phi_p] OR lag-roots [lambda_1..lambda_p]
%   ma_in    — MA parameters: either coefficients [theta_1..theta_q] OR lag-roots [lambda_1..lambda_q]
%
% Name-Value pairs (optional):
%   'ParamType' — 'coeffs' (default) or 'roots'.
%                 If 'roots', we interpret:
%                   A(L) = prod_{i=1}^p (1 - lambda_i L)  => 1 - phi_1 L - ... - phi_p L^p
%                   B(L) = prod_{j=1}^q (1 + lambda_j L)  => 1 + theta_1 L + ... + theta_q L^q
%                 Coefficients are then recovered from these lag polynomials.
%   'BurnIn'   — number of burn-in observations to discard (default 500)
%   'Mu'       — unconditional mean mu (default 0)
%
% Outputs:
%   y  — T-by-1 vector of ARMA(p,q) observations
%   e  — T-by-1 vector of shocks e_t used to generate y
%
% Notes:
%   • Stationarity (AR) / invertibility (MA) are the user’s responsibility.
%   • We simulate with burn-in (default 500) from zero initial conditions.

    p = numel(ar_in); q = numel(ma_in);
    ip = inputParser; ip.KeepUnmatched = true;
    addParameter(ip,'ParamType','coeffs');
    addParameter(ip,'BurnIn',500);
    addParameter(ip,'Mu',0);
    parse(ip,varargin{:});

    paramType = validatestring(ip.Results.ParamType, {'coeffs','roots'});
    B = ip.Results.BurnIn; mu = ip.Results.Mu;

    % Determine phi and theta
    switch paramType
        case 'coeffs'
            phi = ar_in(:).';            % row
            theta = ma_in(:).';
        case 'roots'
            % Build lag polynomials and read off implied coefficients.
            if p>0
                poly_ar = 1; % A(L)
                for i=1:p
                    poly_ar = conv(poly_ar, [1, -ar_in(i)]); % (1 - lambda_i L)
                end
                phi = -poly_ar(2:end);   % A(L) = 1 - phi_1 L - ... - phi_p L^p
            else
                phi = [];
            end
            if q>0
                poly_ma = 1; % B(L)
                for j=1:q
                    poly_ma = conv(poly_ma, [1, ma_in(j)]); % (1 + lambda_j L)
                end
                theta = poly_ma(2:end); % B(L) = 1 + theta_1 L + ... + theta_q L^q
            else
                theta = [];
            end
    end

    % Sanity: warn if (approx) nonstationary / noninvertible
    if ~isempty(phi)
        A = [1, -phi(:).'];
        rr = roots(A);
        if any(abs(rr) <= 1)
            warning('AR polynomial has roots at or inside unit circle; process may be nonstationary.');
        end
    end
    if ~isempty(theta)
        Bpoly = [1, theta(:).'];
        rr = roots(Bpoly);
        if any(abs(rr) <= 1)
            warning('MA polynomial has roots at or inside unit circle; process may be noninvertible.');
        end
    end

    % Simulation with burn-in
    TT = T + B; p = numel(phi); q = numel(theta);
    e = sqrt(sigma2) * randn(TT,1);
    y = zeros(TT,1);
    if mu ~= 0
        % Work with deviations from mu for numerical stability
        x = zeros(TT,1);  % x_t = y_t - mu
        for t = 1:TT
            accAR = 0; accMA = 0;
            for i=1:p
                if t-i >= 1, accAR = accAR + phi(i) * x(t-i); end
            end
            for j=1:q
                if t-j >= 1, accMA = accMA + theta(j) * e(t-j); end
            end
            x(t) = accAR + e(t) + accMA;
        end
        y = x + mu;
    else
        for t = 1:TT
            accAR = 0; accMA = 0;
            for i=1:p
                if t-i >= 1, accAR = accAR + phi(i) * y(t-i); end
            end
            for j=1:q
                if t-j >= 1, accMA = accMA + theta(j) * e(t-j); end
            end
            y(t) = accAR + e(t) + accMA;
        end
    end

    % Drop burn-in
    y = y(B+1:end);
    e = e(B+1:end);
end
