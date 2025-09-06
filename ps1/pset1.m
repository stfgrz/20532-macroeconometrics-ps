%% Setup
clear; clc; close all;
rng(42);                           % reproducibility

%% EXERCISE 1: AR(1) by loop vs filter (same shocks)
phi   = 0.4;  T = 500;  sigma2 = 0.2;
e     = sqrt(sigma2) * randn(T,1);

x = zeros(T,1);                    % loop
for t = 2:T
    x(t) = phi*x(t-1) + e(t);
end
y = filter(1, [1 -phi], e);        % filter

% equality check
fprintf('Ex1: max |x - y| = %.3g\n', max(abs(x-y)));

tiledlayout(1,2,'TileSpacing','compact');
nexttile; plot(x); title('AR(1) – loop'); grid on;
nexttile; plot(y); title('AR(1) – filter'); grid on;

%% EXERCISE 1C: ACFs
tiledlayout(1,2,'TileSpacing','compact'); 
nexttile; autocorr(x,"NumLags",20); title('ACF loop');
nexttile; autocorr(y,"NumLags",20); title('ACF filter');

%% EXERCISE 2: AR(1) with mean μ and burn-in
phi = 0.6; mu = 3; sigma2 = 0.4; T = 500; burn = 50;
c   = (1 - phi)*mu;                         % ensures E[y_t]=mu
e   = sqrt(sigma2)*randn(T,1);
y   = filter(1,[1 -phi], c + e);            % y_t = c + phi y_{t-1} + e_t
y_final = y(burn+1:end);

tiledlayout(1,2,'TileSpacing','compact');
nexttile; plot(y);       title('AR(1), mean 3, initial y_1=0'); grid on;
nexttile; plot(y_final); title('After burn-in (50)'); grid on;

%% EXERCISE 3: MA(1) by loop vs filter (consistent sign)
theta = 0.3;  T = 500;  sigma2 = 0.3;
e     = sqrt(sigma2)*randn(T,1);

x = zeros(T,1);                         % loop: x_t = e_t + theta e_{t-1}
for t = 2:T, x(t) = e(t) + theta*e(t-1); end
y = filter([1 theta], 1, e);            % same convention

fprintf('Ex3: max |x - y| = %.3g\n', max(abs(x-y)));

t = tiledlayout(2,2,'TileSpacing','compact');
ax1 = nexttile; autocorr(x,"NumLags",20); title('ACF loop');
ax2 = nexttile; autocorr(y,"NumLags",20); title('ACF filter');
linkaxes([ax1 ax2],'y');
nexttile; plot(x); title('MA(1) – loop'); grid on;
nexttile; plot(y); title('MA(1) – filter'); grid on;

%% EXERCISE 4: MA(2) ACF and AR(2) PACF (no external arma_generator)
T = 500; sigma2 = 0.4; e = sqrt(sigma2)*randn(T,1);
theta = [0.4 0.6];           % MA(2): e_t + θ1 e_{t-1} + θ2 e_{t-2}
y = filter([1 theta], 1, e);
phi   = [0.4 0.6];           % AR(2): x_t = φ1 x_{t-1} + φ2 x_{t-2} + e_t
x = filter(1, [1 -phi], e);

tiledlayout(2,1,'TileSpacing','compact');
nexttile; autocorr(y,"NumLags",20);  title('ACF of MA(2)');
nexttile; parcorr(x,"NumLags",20);   title('PACF of AR(2)');

%% EXERCISE 5: Monte Carlo for AR(1) OLS (vectorized)
phi = 0.4; T = 250; n = 1000; sigma2 = 1;
E  = sqrt(sigma2)*randn(T,n);                 % innovations matrix T×n
Y  = filter(1,[1 -phi],E);                    % each column is a path
X  = Y(1:end-1,:);  Z = Y(2:end,:);          % lag / current
phihat = sum(X.*Z,1) ./ sum(X.^2,1);

figure; histogram(phihat,'Normalization','pdf');
title('OLS for AR(1), T=250'); xlabel('\phî'); ylabel('pdf');

fprintf('Ex5: mean(phî)=%.4f, sd(phî)=%.4f\n', mean(phihat), std(phihat));

%% EXERCISE 5b: Size of t-test for H0: \phi = 0.4
res   = Z - X.*phihat;                        % implicit expansion
sigma = sqrt( sum(res.^2,1) ./ (T-2) );       % df = (T-1) - 1
se    = sigma ./ sqrt( sum(X.^2,1) );
tstat = (phihat - phi) ./ se;
tcrit = tinv(0.975, T-2);
rej   = mean(abs(tstat) > tcrit);
fprintf('Ex5b: rejection frequency at 5%% (H0 true) = %.3f\n', rej);

figure; histogram(tstat,'Normalization','pdf');
title('t-statistics under H0: \phi=0.4'); xlabel('t'); ylabel('pdf');

%% EXERCISE 6: Near-unit-root, different T
phi = 0.9; Ts = [50 100 200 1000]; n = 1000; sigma2 = 1;
tiledlayout(2,2,'TileSpacing','compact');
for idx = 1:numel(Ts)
    T = Ts(idx);
    E = sqrt(sigma2)*randn(T,n);
    Y = filter(1,[1 -phi],E);
    X = Y(1:end-1,:); Z = Y(2:end,:);
    phihat = sum(X.*Z,1) ./ sum(X.^2,1);
    nexttile; histogram(phihat,'Normalization','pdf');
    title(sprintf('OLS \\phî, T=%d',T)); xlabel('\phî'); ylabel('pdf');
    fprintf('Ex6 T=%d: mean=%.4f, sd=%.4f\n', T, mean(phihat), std(phihat));
end

%% EXERCISE 7: OLS on MA(1) (biased)
theta = 0.6; Ts = [250 500 1000 10000]; n = 1000; sigma2 = 1;
tiledlayout(2,2,'TileSpacing','compact'); sgtitle('OLS on MA(1) is biased');
for idx = 1:numel(Ts)
    T = Ts(idx);
    E = sqrt(sigma2)*randn(T,n);
    X = filter([1 theta],1,E);              % MA(1)
    L = X(1:end-1,:); C = X(2:end,:);
    thetahat = sum(L.*C,1) ./ sum(L.^2,1);  % OLS on lagged regressor
    nexttile; histogram(thetahat,'Normalization','pdf');
    title(sprintf('\\thetâ via OLS, T=%d',T)); xlabel('\thetâ'); ylabel('pdf');
    fprintf('Ex7 T=%d: mean=%.4f, sd=%.4f\n', T, mean(thetahat), std(thetahat));
end