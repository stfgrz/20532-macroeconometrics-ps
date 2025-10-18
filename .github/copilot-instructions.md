# 20532 Macroeconometrics - AI Coding Guide

## Project Architecture

This is a MATLAB-based macroeconometrics coursework repository (Academic Year 2025/2026) focused on empirical time series modeling, estimation, and inference. Each problem set (`ps1/`, `ps2/`, etc.) is self-contained with its main script (`psetN.m`), data, and outputs.

## Key Patterns & Conventions

### File Structure
- **Entry point**: `psN/psetN.m` - main script for each problem set
- **Data handling**: `psN/Data/` for input datasets (may have hardcoded Windows paths that need adjustment)
- **Output organization**: `psN/output/` for figures (`.png`, vector exports) and tables (`.csv`)
- **LaTeX writeups**: `psN/20532_Macroeconometrics___Problem_Set_N/main.tex` with custom `SimpleStef.tex` styling

### MATLAB Coding Standards

#### Reproducible Setup Pattern (appears in every `psetN.m`):
```matlab
clear; clc; close all; format compact
outdir = fullfile(pwd,'psN/output');
if ~exist(outdir,'dir'), mkdir(outdir); end
rng(20532,'twister');  % Course-specific seed for reproducibility
```

#### Figure Styling (consistent across all problem sets):
```matlab
set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
exportFig = @(fh, name) exportgraphics(fh, fullfile(outdir, name), 'ContentType','vector');
```

#### Function Organization
- **Helper functions**: Defined inline at bottom of each `psetN.m` file (not separate `.m` files)
- **Anonymous functions**: Used for simple operations (`ols = @(y,x) deal((x'*x)\(x'*y), ...)`)
- **Common patterns**: Time series simulation, OLS estimation with diagnostics, Monte Carlo loops

### Econometric Workflow Patterns

#### Time Series Simulation
```matlab
% AR(1) simulation with two methods (loop vs filter) - comparison pattern
Y_loop = simulate_ar1_loop(T, phi, sigma2, mu, Y0, eps);
Y_filt = simulate_ar1_filter(T, phi, sigma2, mu, Y0, eps);
max_abs_diff = max(abs(Y_loop - Y_filt));  % Numerical verification
```

#### Monte Carlo Structure
```matlab
R = 5000;  % Standard replication count
for r=1:R
    % Generate data, estimate, store results
    phi_hat(r) = estimate_something(...);
end
% Summarize with mean, std, rejection rates
```

#### OLS with Full Diagnostics
Standard pattern includes: coefficients, standard errors, t-stats, R², residuals, Durbin-Watson, lag correlation.

### Data Handling Conventions

- **Path issues**: Some scripts have hardcoded Windows paths (e.g., `C:\Users\matte\...`) - update to relative paths
- **File types**: Mix of `.csv` and `.xlsx` for datasets
- **Variable naming**: Consistent with econometric notation (`log_GDP`, `phi_hat`, etc.)

### Required Toolboxes
- Econometrics Toolbox (for `adftest`, `estimate`, etc.)
- Statistics and Machine Learning Toolbox
- Optimization Toolbox (optional)

## Development Workflow

### Running Problem Sets
```matlab
% From MATLAB root directory
addpath(genpath(pwd));  % Add all subdirectories to path
run('ps1/pset1.m');     % Execute specific problem set
```

### Output Verification
- Figures saved as vector graphics (`.png`) in `psN/output/`
- Summary tables as `.csv` files with standardized naming (`exN_summary.csv`)
- Console output shows key statistics and verification checks

### When Adding New Problem Sets
1. Copy boilerplate setup from existing `psetN.m`
2. Update `outdir` path to match problem set number
3. Maintain consistent function naming (`fn_*` prefix not used here - functions named descriptively)
4. Follow exercise numbering in comments (`%% ===================== Exercise N =====================`)

## Common Issues
- **Path dependencies**: Update hardcoded file paths to relative paths using `fullfile(pwd, ...)`
- **Toolbox dependencies**: Check `ver` in MATLAB for required toolboxes
- **Random seeds**: Always use `rng(20532,'twister')` for course consistency
- **Figure export**: Use `exportFig` helper for consistent vector output format