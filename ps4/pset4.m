%% PS4 — 20532 Macroeconometrics 
% Problem Set 4
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

%% Exercise 4
clear; clc; close all;

% VAR(2): y_t = A1 y_{t-1} + A2 y_{t-2} + u_t
A1 = [0 0; 0.5 7/12];
A2 = [0.4 0.5; 0 0];

n_vars  = size(A1,1);
horizon = 20;

% Companion matrix F = [A1 A2; I 0]
F = [A1, A2; eye(n_vars), zeros(n_vars)];

% Shock matrix for unit shocks to y1 and y2 at t=0 (state is [y; y(-1)])
S0 = [eye(n_vars); zeros(n_vars)];  % (2n x n), columns: shocks to y1, y2

% IRFs of the state: IRF_state(h) = F^h * S0
IRF_state = zeros(2*n_vars, n_vars, horizon+1);
IRF_state(:,:,1) = S0;
for h = 2:horizon+1
    IRF_state(:,:,h) = F * IRF_state(:,:,h-1);
end
% Extract IRFs of current y_t (top block)
IRF_y = IRF_state(1:n_vars, :, :); % (n x n x H+1)

% Plot: y1→shock1, y2→shock1, y1→shock2, y2→shock2
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
nexttile; plot(0:horizon, squeeze(IRF_y(1,1,:)),'LineWidth',2); grid on;
title('IRF of y1 to First Shock');  xlabel('Time Steps'); ylabel('Response');

nexttile; plot(0:horizon, squeeze(IRF_y(2,1,:)),'LineWidth',2); grid on;
title('IRF of y2 to First Shock');  xlabel('Time Steps'); ylabel('Response');

nexttile; plot(0:horizon, squeeze(IRF_y(1,2,:)),'LineWidth',2); grid on;
title('IRF of y1 to Second Shock'); xlabel('Time Steps'); ylabel('Response');

nexttile; plot(0:horizon, squeeze(IRF_y(2,2,:)),'LineWidth',2); grid on;
title('IRF of y2 to Second Shock'); xlabel('Time Steps'); ylabel('Response');

sgtitle('Impulse Response Functions (IRFs)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Exercise 5: Adjusted for Orthogonalized Shocks using Companion Matrix
clear; clc; close all;

% DGP matrices (same as Exercise 4)
A1 = [0 0; 0.5 7/12];
A2 = [0.4 0.5; 0 0];

T            = 250;      % sample size
num_samples  = 1000;     % Monte Carlo replications
num_vars     = 2;
horizon      = 20;
max_lags     = 4;        % Task (c)
nlags_ecm    = 2;        % #lags in ECM (p=2 -> one lag in Δy)
p_var        = 2;        % VAR order used in tasks (a) and generation

rng(2334);

% Storage
irfs_levels        = zeros(num_samples, num_vars, num_vars, horizon+1);
irfs_differences   = zeros(num_samples, num_vars, num_vars, horizon+1);
irfs_lags          = cell(max_lags, 1);
IRFs_VECM          = zeros(num_samples, num_vars, horizon+1, num_vars);

trace_stat_r1      = zeros(num_samples,1);
max_stat_r1        = zeros(num_samples,1);
reject_trace_r1    = zeros(num_samples,1);
reject_max_r1      = zeros(num_samples,1);

% 5% critical values (2 variables, no trend): r=0 (trace) ~ 15.41, r<=1 ~ 3.84 (max often ~ 14.07 for r=0)
% You had specific constants; to preserve scope, keep your r=1 criticals:
critical_value_trace_r1 = 35.19; % your provided constant
critical_value_max_r1   = 22.29; % your provided constant

for i = 1:num_samples
    %% -------- Data Generation from the given VAR(2) --------
    F = [A1, A2; eye(num_vars), zeros(num_vars)];
    shocks = randn(num_vars, T);          % N(0,I)
    state  = zeros(2*num_vars, T+1);
    for t = 2:T+1
        state(:,t) = F * state(:,t-1) + [shocks(:,t-1); zeros(num_vars,1)];
    end
    y_levels = state(1:num_vars, 2:end)'; % T x n

    %% -------- Task (a): IRFs in Levels (estimate VAR(2)) --------
    [redAR, ~, ~, ~, ~, ~, ~] = varestimy(2, y_levels, 1); % redAR(:,:,1)=I, redAR(:,:,2)=-A1, redAR(:,:,3)=-A2
    A1_hat = -redAR(:,:,2); A2_hat = -redAR(:,:,3);
    F_levels = [A1_hat, A2_hat; eye(num_vars), zeros(num_vars)];
    irf_state = zeros(2*num_vars, num_vars, horizon+1);
    irf_state(:,:,1) = [eye(num_vars); zeros(num_vars)];
    for h = 2:horizon+1
        irf_state(:,:,h) = F_levels * irf_state(:,:,h-1);
    end
    irfs_levels(i, :, :, :) = irf_state(1:num_vars, :, :); % store y-block IRFs

    %% -------- Task (b): IRFs in Differences --------
    dy = diff(y_levels);
    [redAR_diff, ~, ~, ~, ~, ~, ~] = varestimy(1, dy, 1);  % VAR(1) for differences
    A1d_hat = -redAR_diff(:,:,2);                          % A1 on Δy
    irf_diff = zeros(num_vars, num_vars, horizon+1);
    irf_diff(:,:,1) = eye(num_vars);
    for h = 2:horizon+1
        irf_diff(:,:,h) = A1d_hat * irf_diff(:,:,h-1);
    end
    % accumulate (Δy IRFs -> level IRFs)
    irfs_differences(i, :, :, :) = cumsum(irf_diff, 3);

    %% -------- Task (c): IRFs for Varying Lags on Differences --------
    for lag = 1:max_lags
        dy = diff(y_levels);
        [redAR_lag, ~, ~, ~, ~, ~, ~] = varestimy(lag, dy, 1);
        A_list = cell(1, lag);
        for j = 1:lag
            A_list{j} = -redAR_lag(:,:,j+1);
        end
        topBlock = concatAR(A_list); % [A1 ... Ap]
        F_lags = [topBlock; eye(num_vars*(lag-1)), zeros(num_vars*(lag-1), num_vars)];
        irf_tmp = zeros(num_vars*lag, num_vars, horizon+1);
        irf_tmp(1:num_vars,:,1) = eye(num_vars);
        for h = 2:horizon+1
            irf_tmp(:,:,h) = F_lags * irf_tmp(:,:,h-1);
        end
        if isempty(irfs_lags{lag})
            irfs_lags{lag} = zeros(num_samples, num_vars, num_vars, horizon+1);
        end
        irfs_lags{lag}(i, :, :, :) = cumsum(irf_tmp(1:num_vars, :, :), 3);
    end

    %% -------- Task (d): IRFs for ECM via Johansen --------
    % Build Δy_t and regressors for p=2: Δy_t = Γ Δy_{t-1} + Π y_{t-1} + ε_t
    Tn = size(y_levels,1);
    dy_all = diff(y_levels);               % (T-1) x n
    ylag   = y_levels(1:end-1,:);         % (T-1) x n

    % Align to use Δy_{t-1} and y_{t-1} -> sample indices 2..T-1 (length T-2)
    Dy_t   = dy_all(2:end, :);
    Dy_lag = dy_all(1:end-1, :);
    Y_lag  = ylag(1:end-1, :);
    Z      = [ones(Tn-2,1), Dy_lag, Y_lag];          % [const, Δy_{t-1}, y_{t-1}]
    B      = Z \ Dy_t;                               % (1+2n) x n
    const  = B(1,:).';
    Gamma  = B(2:1+num_vars,:).';                    % n x n
    Pi     = B(1+num_vars+1:end,:).';                % n x n

    % Johansen statistics via canonical correlations (no trend; intercept partialed out)
    [lambda, trStat, mxStat] = johansenStats(dy_all, y_levels, 1); % p=2 => 1 lag in Δy
    trace_stat_r1(i) = trStat(2);    % at r=1
    max_stat_r1(i)   = mxStat(2);    % at r=1
    reject_trace_r1(i) = trStat(2) > critical_value_trace_r1;
    reject_max_r1(i)   = mxStat(2)   > critical_value_max_r1;

    % Convert to VAR(2): Φ2 = -Γ, Φ1 = I + Γ + Π
    Phi2 = -Gamma;
    Phi1 = eye(num_vars) + Gamma + Pi;

    % IRFs from ECM-implied VAR
    F_ecm = [Phi1, Phi2; eye(num_vars), zeros(num_vars)];
    irf_state_ecm = zeros(2*num_vars, num_vars, horizon+1);
    irf_state_ecm(:,:,1) = [eye(num_vars); zeros(num_vars)];
    for h = 2:horizon+1
        irf_state_ecm(:,:,h) = F_ecm * irf_state_ecm(:,:,h-1);
    end
    IRFs_VECM(i, :, :, :) = irf_state_ecm(1:num_vars, :, :);
end

%% Task (a): Levels - plot mean and 95% pointwise bands
mean_irfs_levels = squeeze(mean(irfs_levels, 1));
percentiles_irfs_levels = prctile(irfs_levels, [2.5, 97.5], 1);

figure; tiledlayout(num_vars, num_vars, 'TileSpacing','compact','Padding','compact');
for iV = 1:num_vars
    for jS = 1:num_vars
        nexttile; hold on;
        plot(0:horizon, squeeze(mean_irfs_levels(iV, jS, :)), 'b-', 'LineWidth', 2);
        plot(0:horizon, squeeze(percentiles_irfs_levels(1, iV, jS, :)), 'r--', 'LineWidth', 1);
        plot(0:horizon, squeeze(percentiles_irfs_levels(2, iV, jS, :)), 'r--', 'LineWidth', 1);
        title(sprintf('Task a: y%d to Shock y%d', iV, jS));
        xlabel('Horizon'); ylabel('Response'); grid on; hold off;
    end
end

%% Task (b): Differences - plot mean and 95% bands (cumulative)
mean_irfs_differences = squeeze(mean(irfs_differences, 1));
percentiles_irfs_differences = prctile(irfs_differences, [2.5, 97.5], 1);

figure; tiledlayout(num_vars, num_vars, 'TileSpacing','compact','Padding','compact');
for iV = 1:num_vars
    for jS = 1:num_vars
        nexttile; hold on;
        plot(0:horizon, squeeze(mean_irfs_differences(iV, jS, :)), 'b-', 'LineWidth', 2);
        plot(0:horizon, squeeze(percentiles_irfs_differences(1, iV, jS, :)), 'r--', 'LineWidth', 1);
        plot(0:horizon, squeeze(percentiles_irfs_differences(2, iV, jS, :)), 'r--', 'LineWidth', 1);
        title(sprintf('Task b: y%d to Shock y%d', iV, jS));
        xlabel('Horizon'); ylabel('Response'); grid on; hold off;
    end
end

%% Task (c): IRFs for varying lags - with bands (per lag)
for lag = 1:max_lags
    if isempty(irfs_lags{lag}), warning('IRFs for Lag %d are not available.', lag); continue; end
    mean_irfs_lags = squeeze(mean(irfs_lags{lag}, 1));
    percentiles_irfs_lags = prctile(irfs_lags{lag}, [2.5, 97.5], 1);

    figure; tiledlayout(num_vars, num_vars, 'TileSpacing','compact','Padding','compact');
    sgtitle(['IRFs for VAR(', num2str(lag), ') in Differences']);
    for shock_idx = 1:num_vars
        for var_idx = 1:num_vars
            nexttile; hold on;
            plot(0:horizon, squeeze(mean_irfs_lags(var_idx, shock_idx, :)), 'b-', 'LineWidth', 2);
            plot(0:horizon, squeeze(percentiles_irfs_lags(1, var_idx, shock_idx, :)), 'r--', 'LineWidth', 1);
            plot(0:horizon, squeeze(percentiles_irfs_lags(2, var_idx, shock_idx, :)), 'r--', 'LineWidth', 1);
            title(sprintf('y%d \x2192 shock y%d (Lag %d)', var_idx, shock_idx, lag));
            xlabel('Horizon'); ylabel('Response'); grid on; hold off;
        end
    end
end

%% Task (c) consolidated (no bands)
colors = lines(max_lags);
figure; tiledlayout(num_vars, num_vars, 'TileSpacing','compact','Padding','compact');
sgtitle('IRFs for Varying Lags (No Confidence Bands)');
for shock_idx = 1:num_vars
    for var_idx = 1:num_vars
        nexttile; hold on;
        for lag = 1:max_lags
            if isempty(irfs_lags{lag}), continue; end
            mean_irfs_lags = squeeze(mean(irfs_lags{lag}, 1));
            plot(0:horizon, squeeze(mean_irfs_lags(var_idx, shock_idx, :)), ...
                 'LineWidth', 2, 'Color', colors(lag, :), 'DisplayName', sprintf('Lag %d', lag));
        end
        title(sprintf('Response of y%d to Shock in y%d', var_idx, shock_idx));
        xlabel('Horizon'); ylabel('Response'); grid on;
        hold off;
    end
end
legend(arrayfun(@(lag) sprintf('Lag %d', lag), 1:max_lags, 'UniformOutput', false), ...
       'NumColumns', max_lags, 'Location', 'southoutside');

%% Task (d): ECM - mean and 95% bands
mean_IRFs   = mean(IRFs_VECM, 1);                % 1 x n x H x n
lower_CI    = prctile(IRFs_VECM, 2.5, 1);
upper_CI    = prctile(IRFs_VECM, 97.5, 1);

figure; time = 0:horizon;
for var = 1:num_vars
    for shock = 1:num_vars
        subplot_idx = (var-1)*num_vars + shock;
        subplot(num_vars, num_vars, subplot_idx); hold on;
        plot(time, squeeze(lower_CI(1, var, :, shock)), '--r', 'LineWidth', 1.5);
        plot(time, squeeze(upper_CI(1, var, :, shock)), '--r', 'LineWidth', 1.5);
        plot(time, squeeze(mean_IRFs(1, var, :, shock)), 'b', 'LineWidth', 2);
        title(sprintf('IRF: Var %d to Shock %d', var, shock));
        xlabel('Horizon'); ylabel('Response'); grid on; hold off;
    end
end
sgtitle('Impulse Response Functions for ECM with 95% Confidence Intervals');

%% Johansen test summaries (optional display)
fprintf('Mean trace stat at r=1: %.2f | Rejection rate (5%% crit=%.2f): %.2f%%\n', ...
    mean(trace_stat_r1), critical_value_trace_r1, 100*mean(reject_trace_r1));
fprintf('Mean max   stat at r=1: %.2f | Rejection rate (5%% crit=%.2f): %.2f%%\n', ...
    mean(max_stat_r1), critical_value_max_r1, 100*mean(reject_max_r1));

%% ------------------------ Functions used --------------------------
function [redAR, sigma, sidui, R2, constant, Yhat, ex] = varestimy(nlags, X, k, exog)
    % OLS estimation of VAR.
    % redAR(:,:,1)=I, redAR(:,:,j+1) = -A_j (so that I - A1 L - A2 L^2 - ...)
    if nargin == 3, exog = []; ex = 'no exogenous in the system'; end
    [~, c] = size(exog);
    aggre2 = X;
    [obs, variab] = size(aggre2);
    regressori = [];
    for i = 1:nlags
        regressori = [regressori, aggre2(nlags-i+1:size(aggre2,1)-i, :)]; %#ok<AGROW>
    end
    if ~isempty(exog), regressori = [exog(nlags+1:end, :), regressori]; end
    if k == 1, regressori = [ones(size(aggre2,1)-nlags,1), regressori]; end

    Y = aggre2(nlags+1:size(aggre2,1), :);
    iXX = inv(regressori' * regressori); %#ok<NASGU>
    be  = regressori \ Y;
    Yhat = regressori * be;
    sidui = Y - Yhat;

    redAR = zeros(variab, variab, nlags+1);
    if k == 1, constant = be(1, :)'; else, constant = 'no constant in the model'; end
    if ~isempty(exog), ex = be(k+1:c+k, :)'; end

    for i = 1:nlags
        redAR(:, :, i+1) = -be((i-1)*variab+k+c+1:(i-1)*variab+variab+k+c, :)';
    end
    redAR(:, :, 1) = eye(variab);

    R2 = zeros(variab,1);
    for i = 1:variab
        R2(i) = 1 - (sum(sidui(:, i).^2) / sum(center(Y(:, i)).^2));
    end
    sigma = (1 / obs) * (sidui' * sidui);
end

function XC = center(X)
    [T, ~] = size(X);
    XC = X - ones(T,1) * (sum(X)/T);
end

function topBlock = concatAR(A_list)
    % Concatenate {A1,...,Ap} horizontally as [A1 ... Ap]
    p = numel(A_list);
    n = size(A_list{1},1);
    topBlock = zeros(n, n*p);
    for j = 1:p
        topBlock(:, (n*(j-1)+1):(n*j)) = A_list{j};
    end
end

function [lambda, traceStats, maxStats] = johansenStats(dy, y, dLags)
    % Johansen stats (no trend): dLags = p-1
    % Residualize Δy_t and y_{t-1} on Δy_{t-1}..Δy_{t-dLags}
    % Build sample to align regressors:
    % We need R0t: residuals of Δy_t on Δy_{t-1..t-dLags}
    %        R1t: residuals of  y_{t-1} on Δy_{t-1..t-dLags}
    T1 = size(y,1); %#ok<NASGU>
    n  = size(y,2);
    Dy = dy;                 % (T1-1) x n
    Y1 = y(1:end-1,:);       % (T1-1) x n
    % Cut to allow dLags lags of Δy:
    Dy_t   = Dy(1+dLags:end, :);
    Ylag   = Y1(1+dLags:end, :);
    Zdy = [];
    for j = 1:dLags
        Zdy = [Zdy, Dy(dLags+1-j:end-j, :)]; %#ok<AGROW>
    end
    % Now Zdy matches Dy_t length
    % Residualize:
    if isempty(Zdy)
        R0 = Dy_t; R1 = Ylag;
    else
        B0 = Zdy \ Dy_t;  R0 = Dy_t - Zdy * B0;
        B1 = Zdy \ Ylag;  R1 = Ylag - Zdy * B1;
    end
    Tuse = size(R0,1);
    S00 = (R0' * R0) / Tuse;
    S11 = (R1' * R1) / Tuse;
    S01 = (R0' * R1) / Tuse;
    S10 = S01';

    % Solve generalized eigenproblem: |λ S11 - S10 S00^{-1} S01|=0
    [G, D] = eig(S10 / S00 * S01, S11); %#ok<ASGLU>
    lambda = sort(real(diag(D)), 'descend');

    % Trace and max statistics for r=0..n-1 -> length n
    traceStats = zeros(n,1);
    maxStats   = zeros(n,1);
    for r = 0:n-1
        traceStats(r+1) = -Tuse * sum(log(1 - lambda(r+1:end)));
        if r < n-1
            maxStats(r+1) = -Tuse * log(1 - lambda(r+1));
        else
            maxStats(r+1) = NaN;
        end
    end
end
