%% 20532 Macroeconometrics | Problem Set 2
%
% ---------------------------------------------------------------
% Author: Stefano Graziosi
% Date: 2025-11-01
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

%% Exercise 1

%% Exercise 2

%% Helper Functions

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

function [rts, coef] = detC_roots(beta)
% Returns roots of det C(z) and the polynomial coefficients.
% det C(z) = (beta^2/(1-beta)) + beta*z - (beta/(1-beta))*z^2
% poly coefficients in descending powers: [z^2, z, const]
    coef = [-(beta/(1-beta)), beta, beta^2/(1-beta)];
    rts  = roots(coef);  % expected ~ [1 ; -beta]
end

function coverageTbl = irf_coverage_table(trueIRF, loIRF, hiIRF)
% Compute fraction of horizons where true IRF is inside [lo, hi] band
% for each (variable, shock) pair.
    [nvar, nshock, H1] = size(trueIRF);
    H = H1 - 1;
    VarNames   = {'x','y'};
    ShockNames = {'eta','eps_unit'}; % eps normalized to unit variance

    rows = [];
    for i = 1:nvar
        for j = 1:nshock
            tr  = squeeze(trueIRF(i,j,:));
            lo  = squeeze(loIRF(i,j,:));
            hi  = squeeze(hiIRF(i,j,:));
            inside = (tr >= lo) & (tr <= hi);
            fracInside = mean(inside);
            countInside = sum(inside);
            rows = [rows; {VarNames{i}, ShockNames{j}, fracInside, countInside, H1}]; %#ok<AGROW>
        end
    end
    coverageTbl = cell2table(rows, ...
        'VariableNames', {'Variable','Shock','FracInside95','CountInside','NumHorizons'});
end

function plot_eps_to_x_diff(trueIRF, IRF_draws, IRF_mean, p, Nmc, T, exportFig)
% Plot mean vs true for x<-eps (top) and the difference with 95% band (bottom).
    H1 = size(trueIRF,3);
    h  = 0:H1-1;

    % Select x (row 1), shock = epsilon (col 2)
    tr   = squeeze(trueIRF(1,2,:));             % (H+1) x 1
    estm = squeeze(IRF_mean(1,2,:));            % (H+1) x 1
    draws = squeeze(IRF_draws(1,2,:,:));        % (H+1) x N
    diffs = draws - tr;                         % implicit expansion

    diff_mean = mean(diffs, 2);
    diff_lo   = prctile(diffs, 2.5, 2);
    diff_hi   = prctile(diffs, 97.5, 2);

    fh = figure('Position',[100 100 820 620]);
    tlo = tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

    % Top: IRF level — estimated mean vs true
    nexttile; hold on; grid on
    plot(h, estm, '-', 'LineWidth', 1.8, 'DisplayName','MC mean (estimated)');
    plot(h, tr,  '--k', 'LineWidth', 1.6, 'DisplayName','True');
    title('$x_t$ response to $\varepsilon/\sqrt{0.8}$ (level)');
    xlabel('Horizon $h$'); ylabel('Response');
    legend('Location','best');

    % Bottom: Difference with 95% band
    nexttile; hold on; grid on
    fill([h, fliplr(h)], [diff_lo' fliplr(diff_hi')], ...
         [0.85 0.9 1.0], 'EdgeColor','none', 'FaceAlpha',0.7, 'DisplayName','95% band (diff)');
    plot(h, diff_mean, '-', 'LineWidth', 1.8, 'DisplayName','Mean(diff)');
    yline(0, '--k', 'HandleVisibility','off');
    title('Estimated $-$ True for $x\leftarrow \varepsilon/\sqrt{0.8}$');
    xlabel('Horizon $h$'); ylabel('Difference');

    title(tlo, sprintf('Exercise 3: ε-shock to x — VAR(%d), N=%d, T=%d', p, Nmc, T));
    exportFig(fh, sprintf('3_irf_eps_to_x_diff_VAR%d_N%d_T%d.pdf', p, Nmc, T));
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
    
    xline([-zcrit, zcrit], '--', 'HandleVisibility','off');

    yl = ylim;
    text(-zcrit, yl(1), '$-z_{0.975}$', ...
        'Interpreter','latex', 'HorizontalAlignment','left', 'VerticalAlignment','bottom');
    text( zcrit, yl(1), '$z_{0.975}$', ...
        'Interpreter','latex', 'HorizontalAlignment','right', 'VerticalAlignment','bottom');
    
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