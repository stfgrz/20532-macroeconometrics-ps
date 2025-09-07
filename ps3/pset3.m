%% Load the dataset
file_path  = "C:\Users\matte\Documents\ESS\third semester\macroeconometrics\ps 3\data_ps3.xlsx"; % <-- insert your own path 
data = readtable(file_path, 'Sheet', 'monetary_shock', 'VariableNamingRule', 'preserve');
rng(42);

% Extract variables
log_GDP = data.("LOG(GDP)");
log_P   = data.("LOG(P)");
FFR     = data.FFR;

%% ADF Tests for Stationarity (levels)
% Use a trend in levels; DF regression type matters for interpretation
[h_log_GDP, p_log_GDP] = adftest(log_GDP, 'model','TS');
[h_log_P,   p_log_P  ] = adftest(log_P,   'model','TS');
[h_FFR,     p_log_FFR] = adftest(FFR,     'model','TS');

disp('ADF Test Results on Levels (trend included):');
fprintf('LOG(GDP): Stationary: %d, p-value: %.4f\n', h_log_GDP, p_log_GDP);
fprintf('LOG(P):   Stationary: %d, p-value: %.4f\n', h_log_P,   p_log_P);
fprintf('FFR:      Stationary: %d, p-value: %.4f\n', h_FFR,     p_log_FFR);

%% Transformations for Stationarity
log_GDP_diff = diff(log_GDP);
log_P_diff   = diff(log_P);

% Detrend FFR with a linear trend (keeps mean; removes linear time)
FFR_detrended = detrend(FFR, 'linear');

% ADF on transformed data (no drift/trend)
[h_log_GDP_diff, p_log_GDP_diff] = adftest(log_GDP_diff, 'model','AR');
[h_log_P_diff,   p_log_P_diff  ] = adftest(log_P_diff,   'model','AR');
[h_FFR_detrended, p_FFR_detrended] = adftest(FFR_detrended, 'model','AR');

disp('ADF Test Results on Transformed Series:');
fprintf('ΔLOG(GDP): Stationary: %d, p-value: %.4f\n', h_log_GDP_diff, p_log_GDP_diff);
fprintf('ΔLOG(P):   Stationary: %d, p-value: %.4f\n', h_log_P_diff,   p_log_P_diff);
fprintf('FFR detr.: Stationary: %d, p-value: %.4f\n', h_FFR_detrended, p_FFR_detrended);

% Align lengths: differences are T-1, align detrended FFR accordingly
Y_transformed = [log_GDP_diff, log_P_diff, FFR_detrended(2:end)];
variable_names = {'Change in log(GDP)', 'Change in log(Price Level)', 'Detrended FFR'};

%% Determine Optimal Lag Length (BIC)
num_periods = 40;                 % IRF horizon (0..40)
N = size(Y_transformed, 2);       % # variables in VAR
T = size(Y_transformed, 1);       % # observations

lags_to_test = [1, 2, 4];
bic_values = zeros(length(lags_to_test), 1);

for idx = 1:length(lags_to_test)
    p = lags_to_test(idx);
    mdl = varm(N, p);
    Est = estimate(mdl, Y_transformed);
    s = summarize(Est);
    bic_values(idx) = s.BIC;
end

[~, best_bic_idx] = min(bic_values);
best_lag = lags_to_test(best_bic_idx);
fprintf('Best model chosen based on BIC: VAR(%d)\n', best_lag);
nlags  = best_lag;

%% Estimate VAR Model
EstMdl = estimate(varm(N, nlags), Y_transformed);

% Residuals and covariance
epsilon_hat = infer(EstMdl, Y_transformed);
Sigma_u     = EstMdl.Covariance;

% (Optional) check positive definiteness
[~, pchol] = chol(Sigma_u);
assert(pchol==0, 'Residual covariance not positive definite.');

%% Bootstrap loop (method one)
K = 1000;
selected_horizons = [1, 2, 3, 4, 8, 12, 40]; % horizons for FEVD reporting
G_monetary_idx = 3;                          % index for monetary policy shock (FFR)
H = num_periods + 1;                         % include impact (0..num_periods)

IRFs_bootstrap  = zeros(N, H, K);            % responses of all vars to monetary shock
fevd_bootstrap  = zeros(N, length(selected_horizons), K);

Astack = stackAR(EstMdl);                    % N x (N*nlags) stacked AR
cvec   = EstMdl.Constant(:).';               % 1 x N constant
Tadj   = size(epsilon_hat,1);

for k = 1:K
    % Resample reduced-form residuals i.i.d.
    idx = randi(Tadj, Tadj, 1);
    eps_tilde = epsilon_hat(idx, :);

    % Simulate synthetic series under the estimated VAR(p)
    Ytil = simulateVAR(Y_transformed, nlags, cvec, Astack, eps_tilde);

    % Re-estimate the VAR and compute IRFs/FEVD
    Est_tilde = estimate(varm(N, nlags), Ytil);

    % Orthogonalized (Cholesky) IRFs consistent with FEVD
    IRFt = irf(Est_tilde, 'NumObs', H, 'Method','orthogonalized');  % H x N x N
    IRFs_bootstrap(:,:,k) = squeeze(IRFt(:, :, G_monetary_idx)).';  % N x H

    % FEVD in percent for selected horizons
    fevd_all = fevd(Est_tilde, 'NumObs', max(selected_horizons), 'Method','orthogonalized'); % H* x N x N
    fevd_bootstrap(:,:,k) = permute(fevd_all(selected_horizons, :, G_monetary_idx), [2 1 3]) * 100;
end

%% Variance Decomposition Table for First Two Variables
fevd_mean = mean(fevd_bootstrap, 3);

horizons = selected_horizons(:);
variable1_mean = fevd_mean(1, :)';
variable2_mean = fevd_mean(2, :)';

Variance_Decomposition_Table = table(horizons, variable1_mean, variable2_mean, ...
    'VariableNames', {'Horizon', variable_names{1}, variable_names{2}});

disp('Mean Variance Decomposition (in percent) for Selected Horizons:');
disp(Variance_Decomposition_Table);

%% Mean and 95% CIs for IRFs 
IRF_mean  = mean(IRFs_bootstrap, 3);
IRF_lo    = prctile(IRFs_bootstrap,  2.5, 3);
IRF_hi    = prctile(IRFs_bootstrap, 97.5, 3);

[~, numPeriodsInIRF] = size(IRF_mean);

tiledlayout(N,1,'TileSpacing','compact','Padding','compact');
for i = 1:N
    nexttile;
    plot(0:numPeriodsInIRF-1, IRF_mean(i,:), 'b', 'LineWidth', 2); hold on;
    plot(0:numPeriodsInIRF-1, IRF_lo(i,:), 'r--');
    plot(0:numPeriodsInIRF-1, IRF_hi(i,:), 'r--');
    title(['Impulse Response of ', variable_names{i}, ' to Monetary Policy Shock']);
    xlabel('Periods'); ylabel('Response'); grid on; hold off;
    legend('Mean IRF','2.5% CI','97.5% CI','Location','best');
end
sgtitle('Impulse Responses to Monetary Policy Shock with 95% Confidence Intervals');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Exercise 2

filename = "C:\Users\matte\Documents\ESS\third semester\macroeconometrics\ps 3\data_ps3.xlsx"; % <-- insert your own path
sheet = 2;
data2 = readtable(filename, 'Sheet', sheet, 'ReadVariableNames', false);
data2.Properties.VariableNames{2} = 'Productivity';
data2.Properties.VariableNames{3} = 'Hours';

% Log-transform and first-difference
logProductivity   = log(data2.Productivity);
logHours          = log(data2.Hours);
dLogProductivity  = diff(logProductivity) * 100; % growth rate
dLogHours         = diff(logHours)        * 100; % growth rate
Y = [dLogProductivity, dLogHours];

[obs, N2] = size(Y); %#ok<ASGLU>
nlags = 4;

% Estimate VAR(4)
model2 = varm(N2, nlags);
Est2   = estimate(model2, Y);

% Residuals and reduced-form covariance
residuals2 = infer(Est2, Y);
cov2       = cov(residuals2);

% Structural impact via Cholesky
P = chol(cov2, 'lower');

% Companion matrix for VAR(p)
Phi = companionMatrix(Est2);  % (N*nlags) x (N*nlags)

% Long-run impact matrix and restriction H(1,2)=0
A1_to_p = sum(cat(3, Est2.AR{:}), 3);     % N x N
LR = inv(eye(N2) - A1_to_p);              % long-run sum of MA coefficients
% impose H(1,2)=0 by adjusting P(1,2)
P(1, 2) = -(LR(1,2) * P(2,2)) / LR(1,1);
H = LR * P;
assert(abs(H(1,2)) < 1e-8, 'Long-run restriction H(1,2)=0 not satisfied!');

%% Bootstrap Cumulative IRFs
horizon = 12;
num_bootstrap = 1000;
IRFs_tech_cum_bootstrap    = zeros(N2, horizon, num_bootstrap);
IRFs_nonTech_cum_bootstrap = zeros(N2, horizon, num_bootstrap);

Astack2 = stackAR(Est2);
cvec2   = Est2.Constant(:).';
Tadj2   = size(residuals2,1);

for k = 1:num_bootstrap
    % Resample residuals
    idx = randi(Tadj2, Tadj2, 1);
    eps_b = residuals2(idx, :);

    % Simulate synthetic series with preserved initial lags
    Yb = simulateVAR(Y, nlags, cvec2, Astack2, eps_b);

    % Re-estimate VAR on bootstrap data
    Est_b = estimate(varm(N2, nlags), Yb);
    cov_b = Est_b.Covariance;

    % Recompute LR and restriction on bootstrap draw
    LR_b = inv(eye(N2) - sum(cat(3, Est_b.AR{:}), 3));
    P_b  = chol(cov_b, 'lower');
    P_b(1,2) = -(LR_b(1,2) * P_b(2,2)) / LR_b(1,1);

    % IRFs via companion dynamics (structural shocks e1=e_tech, e2=e_nonTech)
    Phi_b = companionMatrix(Est_b);
    IRF_tech    = zeros(N2, horizon);
    IRF_nonTech = zeros(N2, horizon);
    J = [eye(N2); zeros(N2*(nlags-1), N2)];      % selector for first block

    for s = 1:horizon
        PhiPow = Phi_b^(s-1);
        IRF_tech(:,    s) = (J.' * PhiPow * [P_b(:,1); zeros(N2*(nlags-1),1)]);
        IRF_nonTech(:, s) = (J.' * PhiPow * [P_b(:,2); zeros(N2*(nlags-1),1)]);
    end

    IRFs_tech_cum_bootstrap(:, :, k)    = cumsum(IRF_tech,    2);
    IRFs_nonTech_cum_bootstrap(:, :, k) = cumsum(IRF_nonTech, 2);
end

%% Mean and Confidence Intervals
IRF_tech_cum_mean  = mean(IRFs_tech_cum_bootstrap,    3);
IRF_tech_cum_lower = prctile(IRFs_tech_cum_bootstrap, 2.5, 3);
IRF_tech_cum_upper = prctile(IRFs_tech_cum_bootstrap, 97.5, 3);

IRF_nonTech_cum_mean  = mean(IRFs_nonTech_cum_bootstrap,    3);
IRF_nonTech_cum_lower = prctile(IRFs_nonTech_cum_bootstrap, 2.5, 3);
IRF_nonTech_cum_upper = prctile(IRFs_nonTech_cum_bootstrap, 97.5, 3);

% GDP IRFs (sum of first two variables)
IRF_tech_cum_GDP_mean  = IRF_tech_cum_mean(1,:)    + IRF_tech_cum_mean(2,:);
IRF_tech_cum_GDP_lower = IRF_tech_cum_lower(1,:)   + IRF_tech_cum_lower(2,:);
IRF_tech_cum_GDP_upper = IRF_tech_cum_upper(1,:)   + IRF_tech_cum_upper(2,:);

IRF_nonTech_cum_GDP_mean  = IRF_nonTech_cum_mean(1,:)  + IRF_nonTech_cum_mean(2,:);
IRF_nonTech_cum_GDP_lower = IRF_nonTech_cum_lower(1,:) + IRF_nonTech_cum_lower(2,:);
IRF_nonTech_cum_GDP_upper = IRF_nonTech_cum_upper(1,:) + IRF_nonTech_cum_upper(2,:);

%% Plot Bootstrap Results with Reordered Graphs
figure; tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

% Productivity response to tech shock
nexttile;
plot(0:horizon-1, IRF_tech_cum_mean(1,:), 'b', 'LineWidth', 1.5); hold on;
plot(0:horizon-1, IRF_tech_cum_lower(1,:), 'r--');
plot(0:horizon-1, IRF_tech_cum_upper(1,:), 'r--');
title('Productivity Response to Tech Shock'); xlabel('Periods'); ylabel('Cumulative Response'); grid on;

% Productivity response to non-tech shock
nexttile;
plot(0:horizon-1, IRF_nonTech_cum_mean(1,:), 'b', 'LineWidth', 1.5); hold on;
plot(0:horizon-1, IRF_nonTech_cum_lower(1,:), 'r--');
plot(0:horizon-1, IRF_nonTech_cum_upper(1,:), 'r--');
title('Productivity Response to Non-Tech Shock'); xlabel('Periods'); ylabel('Cumulative Response'); grid on;

% GDP response to tech shock
nexttile;
plot(0:horizon-1, IRF_tech_cum_GDP_mean, 'b', 'LineWidth', 1.5); hold on;
plot(0:horizon-1, IRF_tech_cum_GDP_lower, 'r--');
plot(0:horizon-1, IRF_tech_cum_GDP_upper, 'r--');
title('GDP Response to Tech Shock'); xlabel('Periods'); ylabel('Cumulative Response'); grid on;

% GDP response to non-tech shock
nexttile;
plot(0:horizon-1, IRF_nonTech_cum_GDP_mean, 'b', 'LineWidth', 1.5); hold on;
plot(0:horizon-1, IRF_nonTech_cum_GDP_lower, 'r--');
plot(0:horizon-1, IRF_nonTech_cum_GDP_upper, 'r--');
title('GDP Response to Non-Tech Shock'); xlabel('Periods'); ylabel('Cumulative Response'); grid on;

% Hours response to tech shock
nexttile;
plot(0:horizon-1, IRF_tech_cum_mean(2,:), 'b', 'LineWidth', 1.5); hold on;
plot(0:horizon-1, IRF_tech_cum_lower(2,:), 'r--');
plot(0:horizon-1, IRF_tech_cum_upper(2,:), 'r--');
title('Hours Response to Tech Shock'); xlabel('Periods'); ylabel('Cumulative Response'); grid on;

% Hours response to non-tech shock
nexttile;
plot(0:horizon-1, IRF_nonTech_cum_mean(2,:), 'b', 'LineWidth', 1.5); hold on;
plot(0:horizon-1, IRF_nonTech_cum_lower(2,:), 'r--');
plot(0:horizon-1, IRF_nonTech_cum_upper(2,:), 'r--');
title('Hours Response to Non-Tech Shock'); xlabel('Periods'); ylabel('Cumulative Response'); grid on;

sgtitle('Bootstrap Cumulative IRFs');

%% ---------- Helpers ----------
function Astack = stackAR(EstMdl)
    % Return [A1 ... Ap] as N x (N*p)
    N = size(EstMdl.Covariance,1);
    p = EstMdl.P;
    Astack = zeros(N, N*p);
    for j = 1:p
        Astack(:, (N*(j-1)+1):(N*j)) = EstMdl.AR{j};
    end
end

function Ysim = simulateVAR(Y, p, cvec, Astack, eps)
    % Simulate VAR(p): Y_t = c + A1 Y_{t-1} + ... + Ap Y_{t-p} + eps_t
    % Preserves first p observations from Y as initial conditions
    [T0, N] = size(Y);
    T = size(eps,1);
    Ysim = zeros(T + p, N);
    Ysim(1:p, :) = Y(1:p, :);

    % Precompute big AR operator
    BigA = Astack; % N x (N*p)
    for t = p+1:p+T
        % vectorize lags [Y_{t-1}; ...; Y_{t-p}]
        L = Ysim(t-(1:p), :)';
        ylags = L(:);
        Ysim(t, :) = cvec + (BigA * ylags).' + eps(t-p, :);
    end
    Ysim = Ysim(p+1:end, :);
end

function Phi = companionMatrix(EstMdl)
    % Companion matrix for VAR(p)
    N = size(EstMdl.Covariance,1);
    p = EstMdl.P;
    top = zeros(N, N*p);
    for j = 1:p
        top(:, (N*(j-1)+1):(N*j)) = EstMdl.AR{j};
    end
    Phi = [top; eye(N*(p-1)), zeros(N*(p-1), N)];
end