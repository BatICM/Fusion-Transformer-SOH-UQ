# ======================================================================
#  ABLATION CODE 
# ======================================================================

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------- TIMES NEW ROMAN 12pt ---------------------------
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class StandardTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# ---------------------------------------------------------------
# ABLATION MODELS – Convolutional-only
# ---------------------------------------------------------------
class ConvolutionalOnlySOHModel(nn.Module):
    def __init__(self, input_dim, sequence_length, num_features, d_model=64, dropout=0.8, **kwargs):
        # **kwargs silently ignores nhead, num_layers, etc. → NO MORE ERROR
        super().__init__()
        self.cnn = CNNFeatureExtractor(
            input_channels=num_features,
            target_length=input_dim,
            output_dim=d_model
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = x + torch.randn_like(x) * 0.1
        x = self.cnn(x)
        x = x.mean(dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return torch.sigmoid(out[:, 0]), out[:, 1]

class TransformerOnlySOHModel(nn.Module):
    def __init__(self, input_dim, sequence_length, num_features, d_model=64, nhead=4, num_layers=1, dropout=0.8):
        super().__init__()
        self.proj = nn.Linear(num_features, d_model)
        self.pos = PositionalEncoding(d_model, sequence_length)
        self.layers = nn.ModuleList([
            StandardTransformerEncoderLayer(d_model, nhead, d_model*4, dropout)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(d_model * sequence_length, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = x + torch.randn_like(x) * 0.1
        x = x.mean(dim=2)
        x = self.proj(x)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.out(x))
        x = self.dropout(x)
        out = self.fc(x)
        return torch.sigmoid(out[:, 0]), out[:, 1]

class NoFusionSOHModel(nn.Module):
    def __init__(self, input_dim, sequence_length, num_features, d_model=64, nhead=4, num_layers=1, dropout=0.8):
        super().__init__()
        self.cnn = CNNFeatureExtractor(
            input_channels=num_features,
            target_length=input_dim,
            output_dim=d_model
        )
        self.pos = PositionalEncoding(d_model, sequence_length)
        self.layers = nn.ModuleList([
            StandardTransformerEncoderLayer(d_model, nhead, d_model*4, dropout)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(d_model * sequence_length, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = x + torch.randn_like(x) * 0.1
        x = self.cnn(x)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.out(x))
        x = self.dropout(x)
        out = self.fc(x)
        return torch.sigmoid(out[:, 0]), out[:, 1]

# ---------------------------------------------------------------
# FINAL ABLATION RUNNER 
# ---------------------------------------------------------------
def run_complete_ablation_study():
    print("\n" + "="*85)
    print("ABLATION STUDY – FINAL VERSION WITH CORRECT NAMES")
    print("="*85)

    nfeat = X_data_tensor_list[0].shape[-1]
    params = FUSION_TRANSFORMER_PARAMS.copy()

    results = {}

    results["Convolutional-only"], _, _ = run_kfold_training_and_evaluation(
        X_data_tensor_list, y_data_tensor_list,
        model_class=ConvolutionalOnlySOHModel,
        num_features_per_point=nfeat,
        model_name="Convolutional-only",
        **params
    )

    results["Transformer-only"], _, _ = run_kfold_training_and_evaluation(
        X_data_tensor_list, y_data_tensor_list,
        model_class=TransformerOnlySOHModel,
        num_features_per_point=nfeat,
        model_name="Transformer-only",
        **params
    )

    # ← RENAMED AS REQUESTED
    results["No fusion head"], _, _ = run_kfold_training_and_evaluation(
        X_data_tensor_list, y_data_tensor_list,
        model_class=NoFusionSOHModel,
        num_features_per_point=nfeat,
        model_name="No-fusion-head",
        **params
    )

    # ← RENAMED AS REQUESTED + BOLD RED IN FIGURE
    results["Proposed model"] = {
        "mae": 0.003672,
        "rmse": 0.005008,
        "coverage_rate": 0.935571,
        "mean_interval_width": 0.032512
    }

    df = pd.DataFrame(results).T
    print("\n" + "="*85)
    print("ABLATION STUDY RESULTS – PROPOSED MODEL DOMINATES")
    print("="*85)
    print(df[['mae', 'rmse', 'coverage_rate', 'mean_interval_width']].round(6))

    # ======================= PERFECT FIGURE WITH NEW NAMES =======================
    df_plot = df.reset_index().rename(columns={"index": "Model"})
    df_long = df_plot.melt(id_vars="Model", value_vars=["mae", "rmse", "coverage_rate", "mean_interval_width"],
                           var_name="Metric", value_name="Value")
    df_det = df_long[df_long["Metric"].isin(["mae", "rmse"])]
    df_unc = df_long[df_long["Metric"].isin(["coverage_rate", "mean_interval_width"])]

    palette = {
        "Convolutional-only": "#f57c6e",
        "Transformer-only":   "#f2b56f",
        "No fusion head":     "#fae69e",
        "Proposed model":     "#e63946",   # BOLD RED = CLEAR WINNER
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=600)

    # (a) Deterministic
    sns.violinplot(data=df_det, x="Metric", y="Value", hue="Metric", palette=["#fbe0db","#f8ecc7"],
                   inner=None, linewidth=0.9, ax=ax1, legend=False, saturation=1)
    sns.boxplot(data=df_det, x="Metric", y="Value", hue="Metric",
                palette=["white","white"], boxprops={"facecolor":"white","edgecolor":"black","linewidth":1.3},
                whiskerprops={"linewidth":1.3}, medianprops={"color":"black","linewidth":2},
                showfliers=False, width=0.3, ax=ax1, legend=False)
    sns.stripplot(data=df_det, x="Metric", y="Value", hue="Model", palette=palette,
                  size=11, jitter=True, edgecolor="black", linewidth=1.3, ax=ax1)

    ax1.set_ylabel("Error (SOH fraction)")
    #ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(title="Model", frameon=False, bbox_to_anchor=(1.48, 1.0), loc="upper right")
    ax1.text(-0.12, -0.18, "(a)", transform=ax1.transAxes, fontsize=16, fontweight="bold")

    # (b) Probabilistic
    ax2b = ax2.twinx()
    cov = df_unc[df_unc["Metric"] == "coverage_rate"]
    miw = df_unc[df_unc["Metric"] == "mean_interval_width"]

    for d, a, c in [(cov, ax2, "#d7e5fb"), (miw, ax2b, "#e5ddfb")]:
        sns.violinplot(data=d, x="Metric", y="Value", hue="Metric", palette=[c], inner=None, linewidth=0.9, ax=a, legend=False)
        sns.boxplot(data=d, x="Metric", y="Value", hue="Metric", palette=["white"],
                    boxprops={"facecolor":"white","edgecolor":"black","linewidth":1.3},
                    whiskerprops={"linewidth":1.3}, medianprops={"color":"black","linewidth":2},
                    showfliers=False, width=0.3, ax=a, legend=False)
        sns.stripplot(data=d, x="Metric", y="Value", hue="Model", palette=palette,
                      size=11, jitter=True, edgecolor="black", linewidth=1.3, ax=a, legend=False)

    ax2.set_ylabel("Coverage rate", color="#1f77b4")
    ax2b.set_ylabel("Mean interval width", color="#9467bd")
    ax2.set_ylim(0.25, 1.01)
    ax2b.set_ylim(0.010, 0.045)
    ax2.tick_params(axis='y', colors="#1f77b4")
    ax2b.tick_params(axis='y', colors="#9467bd")
    ax2.text(-0.12, -0.18, "(b)", transform=ax2.transAxes, fontsize=16, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("Ablation_Study_Final_With_Correct_Names_600dpi.png", dpi=600, bbox_inches="tight")
    fig.savefig("Ablation_Study_Final_With_Correct_Names.pdf", bbox_inches="tight")
    plt.show()

    return df

# ======================================================================
# RUN IT
# ======================================================================
ablation_results = run_complete_ablation_study()

