# Fusion-Enhanced Transformer for Uncertainty-Aware Battery SOH Estimation

This repository implements the workflow used in the manuscript:

**Fusion-Enhanced Transformer Architecture for Uncertainty-Aware State-of-Health Estimation of Lithium-Ion Batteries Using Partial-Voltage Data**.

The pipeline loads lithium-ion battery cycling data, computes smoothed cycle-wise SOH labels, extracts scalar features from a fixed partial-voltage charging window, selects degradation-relevant features, tiles them into pseudo-profiles, and trains a CNN-Transformer model with a probabilistic output head for SOH prediction and uncertainty estimation. The same script also produces publication-style figures, baseline comparisons, voltage-window sensitivity experiments, and ablation studies.

## Overview

The implemented workflow is:

1. Load `.mat` battery data from the Severson fast-charging dataset.
2. Compute cycle-wise SOH from maximum cycle capacity normalized by the average of early valid cycles.
3. Detect and repair anomalous cycles using interpolation, median filtering, and Savitzky-Golay smoothing.
4. Extract partial-voltage charging data from a fixed window, defaulting to **3.45–3.55 V**.
5. Compute scalar features from the interpolated partial-voltage profiles:
   - capacity difference from an averaged early-life reference profile
   - capacity statistics: mean, skewness, kurtosis, AUC
   - voltage statistics: mean, skewness, kurtosis, AUC
6. Perform correlation-based feature selection against SOH.
7. Tile the selected scalar features along a length-100 axis to create pseudo-profiles for CNN processing.
8. Build temporal sequences across consecutive cycles.
9. Train a probabilistic CNN-Transformer with custom fusion attention.
10. Evaluate with deterministic and uncertainty metrics, then generate all major figures.

## Repository Contents

The provided codes have end-to-end Python script that includes:

- data loading and preprocessing
- feature extraction and feature selection
- pseudo-profile construction from scalar features
- model definition and training
- probabilistic evaluation
- baseline benchmarking with conformal prediction
- voltage-window sensitivity analysis
- ablation experiments
- figure generation for manuscript-style plots

## Requirements

Install the following Python packages before running the script:

```bash
pip install numpy scipy pandas matplotlib seaborn scikit-learn pywavelets torch
```

### Main dependencies used by the code

- numpy
- scipy
- pandas
- matplotlib
- seaborn
- pywt / PyWavelets
- torch
- scikit-learn

## Dataset

The script expects a MATLAB file containing the battery dataset:

```python
DATA_PATH = 'C:/Users/machenike/Desktop/matlab_data1_1_10.mat'
```

Before running, update `DATA_PATH` to the location of your local `.mat` file.

The code is written to load `batch1` from the MATLAB file and process the first 10 selected cells by default:

```python
CELLS_TO_ANALYZE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

## How SOH Is Computed

For each cell:

- the reference initial capacity is the average of the maximum capacities from the first few valid cycles
- raw SOH is computed as `max_capacity / initial_capacity`
- suspicious cycles are flagged if they show:
  - abrupt SOH spikes
  - unrealistic capacity drops
  - abnormally short duration
- missing or flagged SOH values are linearly interpolated
- the SOH trajectory is smoothed with:
  - median filtering
  - Savitzky-Golay filtering

Key preprocessing parameters:

```python
SOH_CORRECTION_PARAMS = {
    'initial_capacity_avg_cycles': 5,
    'capacity_drop_ratio_threshold': 0.80,
    'soh_spike_threshold': 0.05,
    'min_cycle_duration_ratio': 0.2
}
```

## Feature Extraction

The implementation focuses on **scalar features** derived from a partial-voltage charging window.

Default feature window:

```python
FEATURE_EXTRACTION_VOLTAGE_RANGE = (3.45, 3.55)
TARGET_LENGTH_PARTIAL_FEATURES = 100
CHARGE_OR_DISCHARGE_FOCUS = 'charge'
```

### Scalar features computed in the script

- `Q_diff_partial_avg`
- `Q_mean`
- `Q_skew`
- `Q_kurtosis`
- `Q_AUC`
- `V_mean`
- `V_skew`
- `V_kurtosis`
- `V_AUC`

### Feature selection

The script uses a correlation-based selection procedure:

1. compute Pearson correlation between each feature and SOH
2. keep features whose absolute correlation exceeds a threshold
3. remove highly collinear features

The code reports the selected feature names and then tiles only those selected scalar features into pseudo-profiles. In the manuscript, the final retained features are described as `Qdiff,partial,avg`, `QAUC`, and `Vskew`.

## Model Architecture

The implemented model class is:

```python
TemporalTransformerWithCNNFusion
```

It contains:

- a 1D CNN feature extractor over tiled scalar pseudo-profiles
- sinusoidal positional encoding
- a stack of custom Transformer encoder layers
- `ImprovedMultiHeadAttention`, which computes per-head importance weights and fuses head outputs
- a probabilistic regression head returning:
  - `mu` via sigmoid
  - `log_var` for predictive variance

### Implemented components

- `CNNFeatureExtractor`
- `PositionalEncoding`
- `ImprovedMultiHeadAttention`
- `CustomTransformerEncoderLayer`
- `TemporalTransformerWithCNNFusion`

### Default training hyperparameters in code

```python
FUSION_TRANSFORMER_PARAMS = {
    'sequence_length': 5,
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'dropout': 0.2,
    'lr': 0.0001,
    'batch_size': 16,
    'num_epochs': 60,
    'patience': 10,
    'n_splits': 5,
}
```

## Loss Function and Uncertainty

The model is trained with a refined Gaussian negative log-likelihood loss with variance regularization:

```python
refined_negative_log_likelihood_loss(mu, log_var, target, ...)
```

The script then computes 95% prediction intervals from the predicted variance and applies a fold-wise sigma calibration step to better match target coverage.

Evaluation includes:

- MAE
- MSE
- RMSE
- R²
- coverage rate
- mean interval width (MIW)

## Baselines

The script includes lightweight baseline models with split conformal prediction:

- Quantile Gradient Boosting
- Lasso
- kNN
- Decision Tree
- Random Forest

These are used for deterministic and probabilistic comparisons against the proposed model.

## Additional Experiments Included

### 1. Voltage-window sensitivity analysis

The script sweeps several voltage windows and compares their impact on:

- MAE
- RMSE
- coverage
- MIW

The manuscript reports that **3.45–3.55 V** is the best-performing window. The code also hardcodes this window as the precomputed best reference in the final sensitivity figure.

### 2. Ablation study

The code defines and evaluates:

- `ConvolutionalOnlySOHModel`
- `TransformerOnlySOHModel`
- `NoFusionSOHModel`
- `Proposed model`

This reproduces the component-wise comparisons discussed in the paper.

## Generated Outputs

Running the script creates multiple manuscript-style figures, including:

- example overview and partial-window sequence plots
- SOH degradation and Q-profile evolution plots
- scalar feature vs SOH correlation heatmap
- probabilistic SOH prediction grids
- probabilistic scatter grids
- k-fold summary boxplots
- baseline comparison plots
- voltage-window sensitivity plots
- ablation study plots

## Running the Code

After updating `DATA_PATH`, run the script as a standard Python file:

```bash
python your_script_name.py
```

The script is organized as a single notebook-style `.py` file with multiple `if __name__ == "__main__":` sections and analysis blocks. In practice, it runs as one long end-to-end workflow.

## Expected Data Shapes

After scalar feature tiling, each cell's feature tensor has shape:

```text
(num_cycles, 100, num_selected_features)
```

Temporal sequence generation then creates model inputs of shape:

```text
(num_samples, sequence_length, 100, num_selected_features)
```

Inside the CNN-Transformer, the CNN converts each cycle pseudo-profile into an embedding, and the Transformer operates across the cycle sequence dimension.

## Reproducibility Tips

To improve reproducibility:

- keep the same random seed (`42`)
- use the same cell subset and voltage window
- preserve the exact preprocessing thresholds
- keep the same cross-validation setup
- run on the same `.mat` dataset structure expected by the loader

The script already sets seeds for NumPy and PyTorch and uses deterministic CuDNN settings where possible.

## Citation

If you use this code, please cite the associated manuscript:

> Fusion-Enhanced Transformer Architecture for Uncertainty-Aware State-of-Health Estimation of Lithium-Ion Batteries Using Partial-Voltage Data.

## Acknowledgment

The manuscript acknowledges support from:

- National Natural Science Foundation of China
- Sichuan Science and Technology Program
- Natural Science Foundation of Xinjiang Uygur Autonomous Region
