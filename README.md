# 20532 Macroeconometrics — Problem Sets (A.Y. 2025/2026)

Repository of macroeconometrics problem sets and code for the academic year 2025/2026.

- Repository: [stfgrz/20532-macroeconometrics-ps](https://github.com/stfgrz/20532-macroeconometrics-ps)
- Primary language: MATLAB
- License: MIT (see [LICENSE](LICENSE))

## Overview

This repository contains scripts, functions, and write-ups for the 20532 Macroeconometrics course problem sets. The focus is on empirical macroeconometrics techniques, including time series modeling, estimation, and inference.

Typical contents per problem set:
- Reproducible scripts to load data, run models, and produce tables/figures
- Helper functions for estimation and plotting
- A short write-up or notes summarizing the approach and results

## Getting started

1. Clone the repository:
   ```bash
   git clone https://github.com/stfgrz/20532-macroeconometrics-ps.git
   cd 20532-macroeconometrics-ps
   ```
2. Open in MATLAB (recommended) or GNU Octave (see Compatibility below).
3. From MATLAB, set the repo root as your working directory and add subfolders to the path if needed:
   ```matlab
   addpath(genpath(pwd));
   ```
4. Run the main script for a given problem set (e.g., `ps01_main.m`) or use the project’s “run all” script if present.

## Requirements

- MATLAB R2022a or newer recommended
- Toolboxes:
  - Econometrics Toolbox
  - Statistics and Machine Learning Toolbox
- Optional:
  - Optimization Toolbox (for some estimation routines)
- Alternative: GNU Octave 8+ with the `io` and `statistics` packages (see Compatibility)

You can check installed toolboxes in MATLAB with:
```matlab
ver
```

## Data

- Large or restricted datasets are not committed. Place raw data files under a `data/` directory (e.g., `data/raw/` and `data/processed/`).
- Scripts will expect standard filenames and paths; consult the header of each `*_main.m` for required inputs.
- If data are public, a download script (e.g., `scripts/download_data.m`) or instructions may be provided.

Data privacy and licensing:
- Do not commit proprietary or confidential datasets.
- If distributing derived data, include a note on the source and license.

## Reproducibility

- Scripts set random seeds where stochastic routines are used (e.g., `rng(20532, 'twister')`).
- Results (tables/figures) are saved to `results/` or `figures/` subfolders with informative filenames and timestamps when applicable.
- Where feasible, outputs are deterministic given the same inputs and environment.

## Project structure (convention)

- `psXX/` Problem set-specific code (e.g., `ps01/`, `ps02/`)
  - `psXX_main.m` Entry point
  - `functions/` Helper functions
  - `figures/` Generated plots
  - `tables/` Generated tables
- `data/` (not tracked if large)
  - `raw/` Original datasets
  - `processed/` Cleaned/intermediate datasets
- `scripts/` Shared utilities (e.g., data download, transformations)
- `results/` Aggregated outputs, logs
- `docs/` Notes or write-ups

Note: Actual folders may differ slightly; consult each problem set’s `*_main.m`.

## How to run

- From MATLAB:
  ```matlab
  % Example for problem set 1
  run('ps01/ps01_main.m');
  ```
- Command-line MATLAB (non-interactive):
  ```bash
  matlab -batch "run('ps01/ps01_main.m')"
  ```
- Expected outputs:
  - Figures in `ps01/figures/`
  - Tables in `ps01/tables/`
  - Logs in `results/` or console

## Compatibility

- MATLAB is the reference environment.
- GNU Octave: many scripts may work, but minor changes could be necessary (e.g., plotting or toolbox-specific APIs).
- If running Octave:
  ```octave
  pkg load io
  pkg load statistics
  ```

## Contributing

- Keep problem set directories self-contained.
- Follow consistent naming: `psXX_*` for scripts, `fn_*` for helper functions.
- Prefer vectorized MATLAB code; document numerical tolerances used in estimation.
- Add brief headers to scripts describing purpose, inputs, and outputs.

## Citation

If you use parts of this repository in academic work, please cite appropriately. At minimum:
- Course: 20532 Macroeconometrics (A.Y. 2025/2026)
- Author/Maintainer: @stfgrz
- Repository: [20532-macroeconometrics-ps](https://github.com/stfgrz/20532-macroeconometrics-ps)
- License: MIT

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contact

- Maintainer: [@stfgrz](https://github.com/stfgrz)
- Issues and questions: please open an issue in the repository.

---
Badges:
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
