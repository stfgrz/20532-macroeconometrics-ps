%% 20532 Macroeconometrics | Problem Set 3
%
% ---------------------------------------------------------------
% Author: Stefano Graziosi
% Date: 2025-11-01
% Copilot Review: 2025-11-05
% ---------------------------------------------------------------

%% Housekeeping & Graphics Style
clear; clc; close all; format compact;

% Output directory setup
outdir = fullfile(pwd,'ps3/output');
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

% Global settings
rng(20532,'twister'); % Set seed for reproducibility

%% Exercise 1: Monetary Shock VAR (Cholesky Identification)

% --- Settings ---
p = 4;      % VAR lag order
H = 40;     % IRF / FEVD horizon
K = 1000;   % Number of bootstrap replications

% --- Data Loading ---
file_ex1 = 'ps3/data/ps3_monetary_shock.csv';
data_ex1 = readtable(file_ex1, 'PreserveVariableNames', true);

Y = [data_ex1.log_gdp, data_ex1.log_p, data_ex1.ffr];
labels = ["log GDP", "log Price Level", "FFR"];
[n_obs, n_vars] = size(Y);

% --- Question (a): Estimate the VAR on original data ---
VAR_orig = estimateVAR(Y, p);
P_orig = identifyCholesky(VAR_orig.SigmaU);
IRF_orig = computeIRF(VAR_orig, P_orig, H);
FEVD_orig = computeFEVD(IRF_orig);

% --- Questions (b, c, d, e): Bootstrap confidence intervals ---
fprintf('Bootstrapping IRFs for Exercise 1: K=%d, VAR(%d), H=%d\n', K, p, H);

IRF_draws = zeros(n_vars, n_vars, H+1, K);
FEVD_draws = zeros(n_vars, n_vars, H+1, K);

Y_initial = Y(1:p, :); % Initial conditions for simulation

tic
parfor k_boot = 1:K
    % (b) Resample residuals
    U_boot = VAR_orig.U(randi(VAR_orig.n_eff, VAR_orig.n_eff, 1), :);
    
    % (c) Simulate new data series
    Y_sim = simulateVAR(VAR_orig.B, Y_initial, U_boot);

    % (d) Re-estimate VAR on simulated data, identify, and compute IRFs/FEVDs
    VAR_boot = estimateVAR(Y_sim, p);
    P_boot = identifyCholesky(VAR_boot.SigmaU);
    IRF_draws(:,:,:,k_boot) = computeIRF(VAR_boot, P_boot, H);
    FEVD_draws(:,:,:,k_boot) = computeFEVD(IRF_draws(:,:,:,k_boot));
end
toc

% --- Question (f): Plot IRFs with confidence bands ---
IRF_bands = prctile(IRF_draws, [2.5, 50, 97.5], 4);

for j_shock = 1:n_vars
    fh = figure('Position',[90 90 880 760]);
    tlo = tiledlayout(n_vars, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
    
    for i_var = 1:n_vars
        nexttile; hold on; grid on;
        
        % Extract IRFs for the current plot
        irf_point = squeeze(IRF_orig(i_var, j_shock, :));
        irf_lo = squeeze(IRF_bands(i_var, j_shock, :, 1));
        irf_med = squeeze(IRF_bands(i_var, j_shock, :, 2));
        irf_hi = squeeze(IRF_bands(i_var, j_shock, :, 3));
        
        % Plot 95% confidence band
        fill([0:H, H:-1:0], [irf_lo', fliplr(irf_hi')], ...
            [0.85 0.9 1.0], 'EdgeColor', 'none', 'FaceAlpha', 0.8, 'DisplayName', '95% Band');
        
        % Plot point estimate and bootstrap median
        plot(0:H, irf_point, '-', 'LineWidth', 1.8, 'DisplayName', 'Estimate');
        plot(0:H, irf_med, '--', 'LineWidth', 1.2, 'DisplayName', 'Bootstrap Median');
        
        yline(0, 'k:', 'HandleVisibility', 'off');
        ylabel(labels(i_var));
        
        if i_var == 1
            title(sprintf('Responses to a "%s" Shock (Cholesky)', labels(j_shock)));
            legend('Location', 'best');
        end
        if i_var == n_vars, xlabel('Horizon (Quarters)'); end
    end
    
    filename = sprintf('ex1_irf_bands_shock_%s.pdf', strrep(lower(labels(j_shock)), ' ', '_'));
    exportFig(fh, filename);
    close(fh);
end

%% Exercise 2: Technology Shocks (Long-Run Identification)

% --- Settings ---
p_gali = 4;
H_gali = 40;
K_gali = 1000;

% --- Data Prep (Galí 1999, Panel A) ---
file_ex2 = 'ps3/data/ps3_technology_shock.csv';
data_ex2 = readtable(file_ex2, 'PreserveVariableNames', true);

% VAR uses growth rates: Δlog(Productivity), Δlog(Hours)
prod_level = log(data_ex2.y_l);
hours_level = log(data_ex2.hours);

Y_gali = [diff(prod_level), diff(hours_level)];
labels_gali = ["Productivity", "Hours"];
[~, n_vars_gali] = size(Y_gali);

% --- Estimate VAR on original data and identify ---
VAR_gali_orig = estimateVAR(Y_gali, p_gali);
P_gali_orig = identifyLongRun(VAR_gali_orig);
IRF_gali_orig = computeIRF(VAR_gali_orig, P_gali_orig, H_gali);

% --- Bootstrap confidence intervals ---
fprintf('Bootstrapping IRFs for Exercise 2: K=%d, VAR(%d), H=%d\n', K_gali, p_gali, H_gali);

IRF_gali_draws = zeros(n_vars_gali, n_vars_gali, H_gali+1, K_gali);
Y_gali_initial = Y_gali(1:p_gali, :);

tic
parfor k_boot = 1:K_gali
    % Resample residuals
    U_boot = VAR_gali_orig.U(randi(VAR_gali_orig.n_eff, VAR_gali_orig.n_eff, 1), :);
    
    % Simulate new data series
    Y_sim = simulateVAR(VAR_gali_orig.B, Y_gali_initial, U_boot);
    
    % Re-estimate VAR, identify, and compute IRFs
    VAR_boot = estimateVAR(Y_sim, p_gali);
    P_boot = identifyLongRun(VAR_boot);
    IRF_gali_draws(:,:,:,k_boot) = computeIRF(VAR_boot, P_boot, H_gali);
end
toc

% --- Transform IRFs to cumulative levels ---
% Function to convert growth rate IRFs to cumulative levels for plotting
cumulate_irfs = @(irf_draws) cumsum(irf_draws, 3);

% Calculate cumulative IRFs for output (productivity + hours)
irf_output_orig = IRF_gali_orig(1,:,:) + IRF_gali_orig(2,:,:);
irf_output_draws = IRF_gali_draws(1,:,:,:) + IRF_gali_draws(2,:,:,:);

% Combine all variables for plotting: [Prod, Hours, Output]
IRF_plot_orig = cumulate_irfs([IRF_gali_orig; irf_output_orig]);
IRF_plot_draws = cumulate_irfs([IRF_gali_draws; irf_output_draws]);

labels_plot_gali = ["Productivity", "Hours", "Output"];
plot_titles_gali = ["Technology Shock", "Non-Technology Shock"];
n_plot_vars = length(labels_plot_gali);

% --- Plotting (Figure 2 Style) ---
IRF_plot_bands = prctile(IRF_plot_draws, [2.5, 97.5], 4);
fh = figure('Position',[50 50 980 760]);
tlo = tiledlayout(n_plot_vars, n_vars_gali, 'Padding', 'compact', 'TileSpacing', 'compact');

for i_var = 1:n_plot_vars
    for j_shock = 1:n_vars_gali
        nexttile; hold on; grid on;
        
        irf_point = squeeze(IRF_plot_orig(i_var, j_shock, :));
        irf_lo = squeeze(IRF_plot_bands(i_var, j_shock, :, 1));
        irf_hi = squeeze(IRF_plot_bands(i_var, j_shock, :, 2));
        
        fill([0:H_gali, H_gali:-1:0], [irf_lo', fliplr(irf_hi')], ...
            [0.85 0.9 1.0], 'EdgeColor','none', 'FaceAlpha', 0.7);
        plot(0:H_gali, irf_point, 'k-', 'LineWidth', 1.6);
        
        yline(0, 'k--');
        title(sprintf('Response of %s', labels_plot_gali(i_var)));
        ylabel('% Deviation');
        if i_var == n_plot_vars, xlabel('Quarters'); end
        if i_var == 1, title(plot_titles_gali(j_shock)); end
    end
end

title(tlo, sprintf('Replication of Galí (1999) Figure 2 | VAR(%d), K=%d', p_gali, K_gali));
exportFig(fh, sprintf('ex2_gali1999_replication_VAR%d_K%d.pdf', p_gali, K_gali));
close(fh);


%% Helper Functions

% --- VAR Estimation & Simulation ---
function Xlags = mlag(X, p)
    % Creates a matrix of p lags of X.
    [n_obs, n_vars] = size(X);
    Xlags = zeros(n_obs, n_vars * p);
    for i = 1:p
        Xlags(i+1:end, (n_vars*(i-1)+1):n_vars*i) = X(1:end-i, :);
    end
end

function model = estimateVAR(Y, p)
    % Estimates a VAR(p) model with an intercept.
    % Y: (n_obs x n_vars) data matrix
    % p: lag order
    [n_obs, n_vars] = size(Y);
    
    X_lags = mlag(Y, p);
    X = [ones(n_obs, 1), X_lags];
    
    Y_trim = Y(p+1:end, :);
    X_trim = X(p+1:end, :);
    
    model.B = X_trim \ Y_trim; % OLS coefficients
    model.U = Y_trim - X_trim * model.B; % Residuals
    
    model.n_eff = size(Y_trim, 1);
    model.k_reg = size(model.B, 1);
    df = model.n_eff - model.k_reg;
    
    model.SigmaU = (model.U' * model.U) / df; % Residual covariance matrix
    
    % Store coefficients in a structured way
    model.c = model.B(1, :).'; % Intercepts (n_vars x 1)
    model.A = reshape(model.B(2:end, :), n_vars, n_vars*p); % [A1, A2, ..., Ap]
    model.A_comp = [model.A; eye(n_vars*(p-1)), zeros(n_vars*(p-1), n_vars)];
    
    % Store other useful info
    model.p = p;
    model.n_vars = n_vars;
end

function Y_sim = simulateVAR(B, Y_initial, U_boot)
    % Simulates data from a VAR model given coefficients and residuals.
    % B: (1+n_vars*p x n_vars) coefficient matrix from estimateVAR
    % Y_initial: (p x n_vars) matrix of initial conditions
    % U_boot: (n_eff x n_vars) matrix of bootstrapped residuals
    
    [p, n_vars] = size(Y_initial);
    n_eff = size(U_boot, 1);
    n_obs = p + n_eff;
    
    Y_sim = zeros(n_obs, n_vars);
    Y_sim(1:p, :) = Y_initial;
    
    for t = (p+1):n_obs
        % Create the regressor row for time t
        lags_flat = reshape(flipud(Y_sim(t-p:t-1, :))', 1, n_vars*p);
        X_row = [1, lags_flat];
        
        % Simulate Y_t
        Y_sim(t, :) = X_row * B + U_boot(t-p, :);
    end
end

% --- Shock Identification ---
function P = identifyCholesky(SigmaU)
    % Computes the Cholesky factor of the residual covariance matrix.
    try
        P = chol(SigmaU, 'lower');
    catch
        % Fallback for matrices that are not perfectly positive definite
        warning('SigmaU is not positive definite. Nudging to nearest SPD matrix.');
        Sig = (SigmaU + SigmaU') / 2;
        [V, D] = eig(Sig);
        D = max(D, 1e-12 * eye(size(SigmaU,1)));
        P = chol(V * D * V', 'lower');
    end
end

function P_BQ = identifyLongRun(VAR_model)
    % Computes the structural impact matrix P using Blanchard-Quah long-run restrictions.
    n = VAR_model.n_vars;
    p = VAR_model.p;
    
    % Sum of VAR coefficients A(1) = sum(A_k)
    A_sum = zeros(n, n);
    for k = 1:p
        A_k = VAR_model.A(:, (n*(k-1)+1):(n*k));
        A_sum = A_sum + A_k;
    end
    
    % Long-run multiplier C(1) = (I - A(1))^{-1}
    C1 = (eye(n) - A_sum) \ eye(n);
    
    % Long-run covariance of structural shocks: Omega = C(1) * Sigma_u * C(1)'
    Omega = C1 * VAR_model.SigmaU * C1';
    
    % Cholesky factor of Omega gives C(1)*P
    C1_P = identifyCholesky(Omega);
    
    % Recover structural impact matrix P = C(1)^{-1} * (C1_P)
    P_BQ = C1 \ C1_P;
end

% --- Innovation Accounting ---
function IRF = computeIRF(VAR_model, P, H)
    % Computes structural impulse responses.
    % VAR_model: A struct from estimateVAR
    % P: (n_vars x n_vars) structural impact matrix
    % H: Horizon for IRFs
    
    [n_comp, ~] = size(VAR_model.A_comp);
    n = VAR_model.n_vars;
    
    IRF = zeros(n, n, H+1);
    J = [eye(n), zeros(n, n_comp - n)]; % Selection matrix
    
    A_pow = eye(n_comp);
    for h = 0:H
        Phi_h = J * A_pow * J'; % Reduced-form MA coefficient
        IRF(:,:,h+1) = Phi_h * P; % Structural IRF
        A_pow = A_pow * VAR_model.A_comp;
    end
end

function FEVD = computeFEVD(IRF)
    % Computes Forecast Error Variance Decomposition from structural IRFs.
    % IRF: (n_vars x n_vars x H+1) array of impulse responses
    
    [n, ~, H1] = size(IRF);
    H = H1 - 1;
    
    FEVD = zeros(n, n, H1);
    
    % Cumulative sum of squared IRFs over horizons
    irf_sq_cumsum = cumsum(IRF.^2, 3);
    
    for h = 1:H1
        total_variance = sum(irf_sq_cumsum(:,:,h), 2); % Sum over shocks for each variable
        if any(total_variance > 0)
            FEVD(:,:,h) = irf_sq_cumsum(:,:,h) ./ total_variance;
        end
    end
end