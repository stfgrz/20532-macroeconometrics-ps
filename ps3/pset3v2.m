%% 20532 Macroeconometrics | Problem Set 3
%
% ---------------------------------------------------------------
% Author: Stefano Graziosi
% Date: 2025-11-01
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

% Global settings & Critical Values
rng(20532,'twister');                               % Set seed for reproducibility

%% Exercise 1

file = '/Users/stefanograziosi/Documents/GitHub/20532-macroeconometrics-ps/ps3/data/ps3_monetary_shock.csv';
table_3a = readtable(file, 'Delimiter', ',', 'PreserveVariableNames', true);  % keeps names as in file

% Access columns:
obs_q1  = table_3a.obs;
log_gdp = table_3a.log_gdp;
log_p   = table_3a.log_p;
ffr     = table_3a.ffr;

% Stack variables and basic settings
Y      = [log_gdp, log_p, ffr];                    % [real activity, prices, policy]
labels = ["log_gdp","log_p","ffr"];                % Ordering used for Cholesky (Enders §5)
p      = 4;                                        % Quarterly default; adjust as needed
H      = 40;                                       % IRF / FEVD horizons
[T, N] = size(Y);

%%% Question (a)
% Estimate the VAR (suppose with a constant) on the $N$ variables in $y_t$ and store the coefficients (with the constants collected in the $(N\times1)$ vector $\hat c$ and the $(N\times N)$ autoregressive coefficients in the matrix $\hat A$). 
% If there are $p$ lags, you can express the VAR($p$) as a VAR(1) with the companion form) and the $(T\times N)$ residuals $\hat\varepsilon$ (note that $T$ is the number of estimated residuals, those obtained once one eliminated the lags lost in the estimation).

Xlag   = mlag(Y, p);                               % [L1(Y) ... Lp(Y)]  (uses your helper)
X      = [ones(T,1), Xlag];                        % Add intercept
Ytrim  = Y(p+1:end, :);
Xtrim  = X(p+1:end, :);

B      = Xtrim \ Ytrim;                            % (1+N*p) x N, columns = equations
U      = Ytrim - Xtrim*B;                          % Reduced-form residuals (T_eff x N)
T_eff  = size(U,1);
k      = size(B,1);
df     = T_eff - k;

SigmaU = (U' * U) / df;                            % Residual covariance (unbiased)

c_hat  = B(1,:).';                                 % N x 1 intercepts
A_hat  = reshape(B(2:end,:), N, N*p);              % N x (N*p) stacked [A1 ... Ap]
A_blk  = cell(p,1);                                % Store Ai blocks for simulation
for i = 1:p
    A_blk{i} = A_hat(:, (N*(i-1)+1) : N*i);        % Ai is N x N
end

% Companion VAR(1) form
Acomp  = [A_hat; eye(N*(p-1)), zeros(N*(p-1), N)]; % (N*p) x (N*p)

% Cholesky factor for orthogonalization (Enders (5.38)-(5.41))
try
    P = chol(SigmaU, 'lower');                     % Σ_u = P P' ; ε_t = P^{-1} u_t ; IRF = Φ_h P
catch
    % In case Σ_u is nearly singular, nudge to nearest SPD
    Sig = (SigmaU + SigmaU')/2;
    [V,D] = eig(Sig);
    D = max(D, 1e-12*eye(N));
    P = chol(V*D*V', 'lower');
end

% Save “original” VAR objects (handy to compare against bootstrap draw)
VAR_orig = struct('p',p,'labels',labels,'c',c_hat,'A',A_hat,'Ablocks',{A_blk}, ...
                  'Acomp',Acomp,'U',U,'SigmaU',SigmaU,'P',P);

%%% Question (b)
% Sample with replacement from the estimated residuals so to form a new series of residuals $\tilde\varepsilon$ of dim $(T\times N)$. 
% One way of doing it is generate $T$ random integers from $1$ to $T$ (for example, call the random draw of integers \texttt{PER}) and then set $\tilde\varepsilon = \hat\varepsilon[\texttt{PER},:]$; don’t use the command \texttt{permute}.

PER         = randi(T_eff, [T_eff, 1]);            % indices 1..T_eff
eps_tilde   = U(PER, :);                           % T_eff x N bootstrapped residuals

%%% Question (c)
% Use the newly generated residuals and the estimated coefficients to construct new series:
%    \[
%        \tilde y_t \;=\; \hat c \;+\; \hat A\,\tilde y_{t-1} \;+\; \tilde\varepsilon_t.
%    \]
%    The starting values are the first values of $y_t$ (in the case of a VAR(1), just $y_1$).

% Initialize with the *actual* first p observations (standard practice).
Ysim        = zeros(T, N);
Ysim(1:p,:) = Y(1:p,:);

for t = (p+1):T
    ysum = zeros(N,1);
    for L = 1:p
        ysum = ysum + A_blk{L} * Ysim(t-L, :)';
    end
    % Align boot residual index so eps_tilde(1,:) corresponds to t=p+1
    Ysim(t,:) = (c_hat + ysum)' + eps_tilde(t-p, :);
end

Ysim_trim = Ysim(p+1:end, :);                      % Align with estimation sample length

%%% Question (d)
% Estimate a VAR on the new series $\tilde y_t$; identify shocks and compute im\-pulse responses and variance decompositions. Store the impulse responses and variance decompositions. 

% Estimation on the simulated data (same p)
Xlag_s  = mlag(Ysim, p);
X_s     = [ones(T,1), Xlag_s];
B_s     = X_s(p+1:end,:) \ Ysim_trim;
U_s     = Ysim_trim - X_s(p+1:end,:)*B_s;
df_s    = size(U_s,1) - size(B_s,1);
SigmaU_s= (U_s' * U_s) / df_s;

c_s     = B_s(1,:).';
A_s     = reshape(B_s(2:end,:), N, N*p);
A_blk_s = cell(p,1);
for i = 1:p
    A_blk_s{i} = A_s(:, (N*(i-1)+1) : N*i);
end
Acomp_s = [A_s; eye(N*(p-1)), zeros(N*(p-1), N)];

% Cholesky identification on simulated VAR
try
    P_s = chol(SigmaU_s, 'lower');
catch
    Sig = (SigmaU_s + SigmaU_s')/2;
    [V,D] = eig(Sig);
    D = max(D, 1e-12*eye(N));
    P_s = chol(V*D*V', 'lower');
end

% Structural IRFs via companion MA: Θ_h = Φ_h P  (Enders §5 “innovation accounting”)  :contentReference[oaicite:8]{index=8}
np    = size(Acomp_s,1);
J     = [eye(N), zeros(N, np-N)];
Phi   = zeros(N,N,H+1);                             % reduced-form MA matrices Φ_0..Φ_H
IRF   = zeros(N,N,H+1);                             % structural IRFs Θ_h

A_pow = eye(np);
for h = 0:H
    Phi(:,:,h+1) = J * A_pow * J';
    IRF(:,:,h+1) = Phi(:,:,h+1) * P_s;             % orthogonalized (unit-variance shocks)
    A_pow = A_pow * Acomp_s;
end

% FEVD: for horizon n, contribution of shock j to var(i) = sum_{s=0}^{n-1} Θ_s(i,j)^2 / total
FEVD = zeros(N,N,H);                                % horizons 1..H (no h=0 FEVD)
for n = 1:H
    C = zeros(N,N);                                 % cumulative squares up to n-1
    for s = 0:(n-1)
        G  = IRF(:,:,s+1);                          % Θ_s
        C  = C + G.^2;                              % elementwise square & accumulate
    end
    for i = 1:N
        denom = sum(C(i,:));
        if denom > 0
            FEVD(i,:,n) = C(i,:) / denom;
        else
            FEVD(i,:,n) = NaN;
        end
    end
end

% Store everything neatly; save to disk for later steps (e,f)
VAR_boot1 = struct( ...
    'p',p,'labels',labels, ...
    'Ysim',Ysim,'Ysim_trim',Ysim_trim, ...
    'B',B_s,'c',c_s,'A',A_s,'Ablocks',{A_blk_s},'Acomp',Acomp_s, ...
    'U',U_s,'SigmaU',SigmaU_s,'P',P_s, ...
    'IRF',IRF,'Phi',Phi,'FEVD',FEVD, ...
    'H',H,'ordering','[log\\_gdp, log\\_p, ffr] (Cholesky)');

save(fullfile(outdir, sprintf('ps3_ex1_bootstrap_draw1_VARp%d_H%d.mat', p, H)), ...
     'VAR_orig','VAR_boot1');

%%% Question (e)
% Repeat steps b. to d. \(K\) times (e.g., \(K=1000\)).

K = 1000;                                        % number of bootstrap draws
if ~exist('IRF_orig','var')
    IRF_orig = var_irf_from_companion(Acomp, P, H);  % baseline IRFs from original VAR
end

IRF_draws = zeros(N, N, H+1, K);                 % store all structural IRFs

% Precompute for speed / consistency
J      = [eye(N), zeros(N, N*(p-1))];
np     = size(Acomp,1);

fprintf('Bootstrapping IRFs: K=%d, VAR(%d), H=%d\n', K, p, H);
tic
for kboot = 1:K
    % ---- (b) Resample residuals row-wise ----
    PER       = randi(T_eff, [T_eff, 1]);
    eps_tilde = U(PER, :);                       % T_eff x N

    % ---- (c) Simulate new series with fixed initial conditions ----
    Ysim        = zeros(T, N);
    Ysim(1:p,:) = Y(1:p,:);                      % use actual first p obs
    for t = (p+1):T
        ysum = zeros(N,1);
        for L = 1:p
            ysum = ysum + A_blk{L} * Ysim(t-L, :)';
        end
        Ysim(t,:) = (c_hat + ysum)' + eps_tilde(t-p, :);
    end
    Ysim_trim = Ysim(p+1:end, :);

    % ---- (d) Re-estimate VAR on Ysim and compute IRFs (Cholesky) ----
    Xlag_s  = mlag(Ysim, p);
    X_s     = [ones(T,1), Xlag_s];
    B_s     = X_s(p+1:end,:) \ Ysim_trim;
    U_s     = Ysim_trim - X_s(p+1:end,:)*B_s;
    df_s    = size(U_s,1) - size(B_s,1);
    SigmaU_s= (U_s' * U_s) / df_s;

    A_s     = reshape(B_s(2:end,:), N, N*p);
    Acomp_s = [A_s; eye(N*(p-1)), zeros(N*(p-1), N)];

    % Robust Cholesky
    try
        P_s = chol(SigmaU_s, 'lower');
    catch
        Sig = (SigmaU_s + SigmaU_s')/2;
        [V,D] = eig(Sig);
        D = max(D, 1e-12*eye(N));
        P_s = chol(V*D*V', 'lower');
    end

    % Structural IRFs via companion MA: Θ_h = Φ_h P_s
    np_s  = size(Acomp_s,1);
    if np_s ~= np
        % (Should not happen unless dimensions change)
        J = [eye(N), zeros(N, np_s-N)];
    end
    A_pow = eye(np_s);
    for h = 0:H
        Phi_h = J * A_pow * J';
        IRF_draws(:,:,h+1,kboot) = Phi_h * P_s;
        A_pow = A_pow * Acomp_s;
    end

    if mod(kboot, max(1, floor(K/10))) == 0
        fprintf('  ... %d/%d draws completed\n', kboot, K);
    end
end
toc

% Save draws to disk for reproducibility / later use
save(fullfile(outdir, sprintf('ps3_ex1_bootstrap_IRFs_VARp%d_H%d_K%d.mat', p, H, K)), ...
     'IRF_draws','IRF_orig','labels','p','H','K');

%%% Question (f)
% At the end you have a set of 1000 impulse responses. Plot the 2.5\% and 97.5\% percentile (command \texttt{prctile}) of that empirical distribution. This is your 95\% confidence interval.

IRF_lo   = prctile(IRF_draws,  2.5, 4);
IRF_hi   = prctile(IRF_draws, 97.5, 4);
IRF_med  = prctile(IRF_draws, 50.0, 4);   % optional: median curve

h = 0:H;

% Plot: one figure per shock; each figure has N panels (responses)
for j = 1:N
    fh  = figure('Position',[90 90 880 760]);
    tlo = tiledlayout(N,1,'Padding','compact','TileSpacing','compact');

    for i = 1:N
        nexttile; hold on; grid on

        lo = squeeze(IRF_lo(i,j,:));
        hi = squeeze(IRF_hi(i,j,:));
        md = squeeze(IRF_med(i,j,:));
        bt = squeeze(IRF_draws(i,j,:,:)); %#ok<NASGU> % (H+1) x K (unused here but handy to inspect)
        est= squeeze(IRF_orig(i,j,:));

        % 95% band
        fill([h, fliplr(h)], [lo' fliplr(hi')], ...
            [0.85 0.9 1.0], 'EdgeColor','none', 'FaceAlpha',0.8, 'DisplayName','95% band');

        % Estimated (original-sample) IRF
        plot(h, est, '-', 'LineWidth', 1.8, 'DisplayName','Estimate');

        % Median bootstrap IRF (optional guide)
        plot(h, md, '--', 'LineWidth', 1.2, 'DisplayName','Bootstrap median');

        yline(0,'k:','HandleVisibility','off');
        ylabel(labels(i));
        if i==1
            title(sprintf('IRFs to shock: %s (Cholesky ordering)', labels(j)));
        end
        if i==N
            xlabel('Horizon h');
        end
        if i==1
            legend('Location','best');
        end
    end

    exportFig(fh, sprintf('ps3_ex1_irf_bands_shock_%s_VARp%d_H%d_K%d.pdf', labels(j), p, H, K));
    close(fh);
end

% Also save the percentile arrays (useful for tables / grading artifacts)
save(fullfile(outdir, sprintf('ps3_ex1_irf_percentiles_VARp%d_H%d_K%d.mat', p, H, K)), ...
     'IRF_lo','IRF_hi','IRF_med','labels','p','H','K');

%% Exercise 2
% Read the paper \emph{``\href{https://www.aeaweb.org/articles?id=10.1257/aer.89.1.249}{Technology, Employment, and the Business Cycle: Do Technology Shocks Explain Aggregate Fluctuations?}''} by Jordi Gal\'{\i} \cite{10.1257/aer.89.1.249}, \emph{American Economic Review}, 89(1), 1999, 249--271.
% Using the dataset in the second sheet of \texttt{data\_ps3.xlsx}, replicate Figure 2 in the paper and compute bootstrapped confidence bands.

file = '/Users/stefanograziosi/Documents/GitHub/20532-macroeconometrics-ps/ps3/data/ps3_technology_shock.csv';
table_3b = readtable(file, 'Delimiter', ',', 'PreserveVariableNames', true);  % keeps names as in file

% -------------------- Settings --------------------
p  = 4;                          % quarterly VAR lags (Galí uses short lags in bivariate model)
H  = 40;                         % horizons (quarters)
K  = 1000;                       % bootstrap replications
rng(20532,'twister');

% -------------------- Data prep (consistent with Galí 1999, Panel A) --------------------
% Inputs are LEVELS: OPHNFB (output per hour), HOURS (hours per capita).
% VAR uses GROWTH RATES: Δlog(OPHNFB), Δlog(HOURS per capita).

lp  = log(y_t(:));        % log productivity level (OPHNFB)
lhc = log(hours(:));      % log hours-per-capita level

yg = diff(lp);            % productivity growth Δlog(Y/H)
hg = diff(lhc);           % labor-input growth Δlog(Hours per capita)

Y = [yg hg];              % columns: [Δprod, Δhours_pc]
[T, n] = size(Y);
assert(n==2, 'Expecting two variables: Δprod and Δhours per capita');


% Align & drop first obs if needed
keep = ~isnan(yg) & ~isnan(hg);
yg = yg(keep); hg = hg(keep);

Y = [yg hg];                     % columns: [Δprod, Δhours]
[T, n] = size(Y);
assert(n==2, 'Expecting two variables: Δprod and Δhours');

% -------------------- Settings --------------------
p  = 4;                          % quarterly VAR lags (Galí uses short lags in bivariate model)
H  = 40;                         % horizons (quarters)
K  = 1000;                       % bootstrap replications
rng(20532,'twister');

% -------------------- Estimate reduced-form VAR --------------------
Xlag  = mlag(Y, p);                      % from your helpers
X     = [ones(T,1), Xlag];
Ytrim = Y(p+1:end, :);
Xtrim = X(p+1:end, :);
Tef   = size(Ytrim,1);

B = Xtrim \ Ytrim;                       % (1 + n*p) x n
U = Ytrim - Xtrim * B;                   % Tef x n residuals
kpar = size(B,1);
SigmaU = (U' * U) / (Tef - kpar);        % n x n residual covariance

% Companion + VAR coefficients A = [A1 ... Ap]
A  = B(2:end,:).';                       % n x (n*p)
Acomp = [A; eye(n*(p-1)), zeros(n*(p-1), n)];

% -------------------- BQ long-run identification --------------------
% Long-run multiplier L = (I - A(1))^{-1}, with A(1) = sum_k A_k
I2 = eye(n);
A1sum = zeros(n);
for k = 1:p
    Ak = A(:, (n*(k-1)+1):(n*k));
    A1sum = A1sum + Ak;
end
Linf = (I2 - A1sum) \ I2;                % (I - sum Ak)^(-1)

% Long-run covariance Ω = L Σu L'
OmegaLR = Linf * SigmaU * Linf';

% Enforce lower-triangular long-run impact (technology is shock #1)
Nlr = chol(OmegaLR, 'lower');            % = L * P  (lower triangular)
P_BQ = Linf \ Nlr;                        % structural impact matrix so that e_t = P ε_t

% -------------------- Structural IRFs --------------------
TH = var_irf_from_companion(Acomp, P_BQ, H);    % from your helpers, TH(:,:,h+1)

% Reduced-form to reported objects (levels = cumulative sums of growth)
h = (0:H)';  H1 = H+1;

irf_prod_g   = squeeze(TH(1,1,:));                 % Δprod to tech
irf_hours_g  = squeeze(TH(2,1,:));                 % Δhours to tech
irf_out_g  = irf_prod_g + irf_hours_g;             % Δlog(Output per capita) = Δlog(Y/H) + Δlog(Hours per capita)


irf_prod_g2  = squeeze(TH(1,2,:));                 % Δprod to nontech
irf_hours_g2 = squeeze(TH(2,2,:));                 % Δhours to nontech
irf_out_g2 = irf_prod_g2 + irf_hours_g2;           % for non-technology shock

% Cumulate to levels (percent deviations)
C_tech_prod  = cumsum(irf_prod_g);
C_tech_out   = cumsum(irf_out_g);
C_tech_hours = cumsum(irf_hours_g);

C_nt_prod    = cumsum(irf_prod_g2);
C_nt_out     = cumsum(irf_out_g2);
C_nt_hours   = cumsum(irf_hours_g2);

% -------------------- Bootstrap bands (residual bootstrap) --------------------
Ctech_prod_draws  = zeros(H1, K);
Ctech_out_draws   = zeros(H1, K);
Ctech_hours_draws = zeros(H1, K);

Cnt_prod_draws    = zeros(H1, K);
Cnt_out_draws     = zeros(H1, K);
Cnt_hours_draws   = zeros(H1, K);

Y0 = Y(1:p, :);                                % starting values

for b = 1:K
    % 1) resample residuals with replacement
    idx = randi(Tef, Tef, 1);
    Ub  = U(idx, :);

    % 2) simulate bootstrap sample using estimated B and resampled residuals
    Yb = zeros(T, n);
    Yb(1:p, :) = Y0;                            % same initials
    for t = (p+1):T
        xrow = [1, reshape(flipud(Yb(t-p:t-1,:)).', 1, n*p)];   % [1 L1(Y) ... Lp(Y)]
        Yb(t,:) = xrow * B + Ub(t-p,:);                          % align with U length
    end

    % 3) re-estimate VAR on Yb
    Xlagb  = mlag(Yb, p);
    Xb     = [ones(T,1), Xlagb];
    Ybtrim = Yb(p+1:end,:);
    Xbtrim = Xb(p+1:end,:);
    Tefb   = size(Ybtrim,1);

    Bb = Xbtrim \ Ybtrim;
    Ub_res = Ybtrim - Xbtrim * Bb;
    kparb  = size(Bb,1);
    SigU_b = (Ub_res' * Ub_res) / (Tefb - kparb);

    Ab = Bb(2:end,:).';

    % long-run pieces
    A1sumb = zeros(n);
    for k2 = 1:p
        Akb = Ab(:, (n*(k2-1)+1):(n*k2));
        A1sumb = A1sumb + Akb;
    end
    Linfb   = (I2 - A1sumb) \ I2;
    Omega_b = Linfb * SigU_b * Linfb';
    Nlr_b   = chol(Omega_b, 'lower');
    P_BQ_b  = Linfb \ Nlr_b;

    % IRFs
    Acomp_b = [Ab; eye(n*(p-1)), zeros(n*(p-1), n)];
    THb = var_irf_from_companion(Acomp_b, P_BQ_b, H);

    prod_g_b   = squeeze(THb(1,1,:));              % to tech
    hours_g_b  = squeeze(THb(2,1,:));
    out_g_b    = prod_g_b + hours_g_b;

    prod_g2_b  = squeeze(THb(1,2,:));              % to nontech
    hours_g2_b = squeeze(THb(2,2,:));
    out_g2_b   = prod_g2_b + hours_g2_b;

    Ctech_prod_draws(:,b)  = cumsum(prod_g_b);
    Ctech_out_draws(:,b)   = cumsum(out_g_b);
    Ctech_hours_draws(:,b) = cumsum(hours_g_b);

    Cnt_prod_draws(:,b)    = cumsum(prod_g2_b);
    Cnt_out_draws(:,b)     = cumsum(out_g2_b);
    Cnt_hours_draws(:,b)   = cumsum(hours_g2_b);
end

% Percentile bands
pct = [2.5 97.5];
[lo_tp, hi_tp] = deal(prctile(Ctech_prod_draws,  pct(1), 2), prctile(Ctech_prod_draws,  pct(2), 2));
[lo_to, hi_to] = deal(prctile(Ctech_out_draws,   pct(1), 2), prctile(Ctech_out_draws,   pct(2), 2));
[lo_th, hi_th] = deal(prctile(Ctech_hours_draws, pct(1), 2), prctile(Ctech_hours_draws, pct(2), 2));

[lo_np, hi_np] = deal(prctile(Cnt_prod_draws,    pct(1), 2), prctile(Cnt_prod_draws,    pct(2), 2));
[lo_no, hi_no] = deal(prctile(Cnt_out_draws,     pct(1), 2), prctile(Cnt_out_draws,     pct(2), 2));
[lo_nh, hi_nh] = deal(prctile(Cnt_hours_draws,   pct(1), 2), prctile(Cnt_hours_draws,   pct(2), 2));

% -------------------- Plot (Figure 2 style) --------------------
fh = figure('Position',[50 50 980 760]);
tlo = tiledlayout(3,2,'Padding','compact','TileSpacing','compact');

plotBand = @(x,lo,hi) fill([x; flipud(x)], [lo; flipud(hi)], [0.85 0.9 1.0], ...
                           'EdgeColor','none', 'FaceAlpha',0.7);

% Row 1: Productivity (level)
nexttile; hold on; grid on
plotBand(h, lo_tp, hi_tp); plot(h, C_tech_prod, 'k-', 'LineWidth',1.6)
title('Technology shock \rightarrow Productivity (level)'); xlabel('Quarters'); ylabel('%')
yline(0,'k--'); 

nexttile; hold on; grid on
plotBand(h, lo_np, hi_np); plot(h, C_nt_prod, 'k-', 'LineWidth',1.6)
title('Non-technology shock \rightarrow Productivity (level)'); xlabel('Quarters'); ylabel('%')
yline(0,'k--');

% Row 2: Output (level)
nexttile; hold on; grid on
plotBand(h, lo_to, hi_to); plot(h, C_tech_out, 'k-', 'LineWidth',1.6)
title('Technology shock \rightarrow Output per capita (level)'); xlabel('Quarters'); ylabel('%')
yline(0,'k--');

nexttile; hold on; grid on
plotBand(h, lo_no, hi_no); plot(h, C_nt_out, 'k-', 'LineWidth',1.6)
title('Non-technology shock \rightarrow Output per capita (level)'); xlabel('Quarters'); ylabel('%')
yline(0,'k--');

% Row 3: Hours (level)
nexttile; hold on; grid on
plotBand(h, lo_th, hi_th); plot(h, C_tech_hours, 'k-', 'LineWidth',1.6)
title('Technology shock \rightarrow Hours per capita (level)'); xlabel('Quarters'); ylabel('%')
yline(0,'k--');

nexttile; hold on; grid on
plotBand(h, lo_nh, hi_nh); plot(h, C_nt_hours, 'k-', 'LineWidth',1.6)
title('Non-technology shock \rightarrow Hours per capita (level)'); xlabel('Quarters'); ylabel('%')
yline(0,'k--');

title(tlo, sprintf('Exercise 2 — BQ VAR(%d): IRFs with 95%% bootstrap bands (K=%d, H=%d)', p, K, H));
exportFig(fh, sprintf('2_gali1999_fig2_VAR%d_K%d_H%d.pdf', p, K, H));
close(fh);

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