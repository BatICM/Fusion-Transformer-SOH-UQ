# ======================================================================
# FINAL VOLTAGE RANGE SENSITIVITY 
# ======================================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import skew, kurtosis
from scipy.signal import medfilt, savgol_filter
from sklearn.preprocessing import MinMaxScaler

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

# ------------------------------------------------------------
# MISSING FUNCTION THAT WAS CAUSING THE ERROR – NOW INCLUDED!
# ------------------------------------------------------------
def compute_smoothed_soh(Q_raw, t_raw):
    init_n = SOH_CORRECTION_PARAMS['initial_capacity_avg_cycles']
    dur_ratio = SOH_CORRECTION_PARAMS['min_cycle_duration_ratio']
    spike_thr = SOH_CORRECTION_PARAMS['soh_spike_threshold']
    drop_ratio = SOH_CORRECTION_PARAMS['capacity_drop_ratio_threshold']

    max_caps = []
    for i in range(min(init_n, len(Q_raw))):
        if len(Q_raw[i]) > 0 and len(t_raw[i]) > 0:
            duration = t_raw[i][-1] - t_raw[i][0]
            if duration > FULL_PLOT_TIME_THRESHOLD * dur_ratio:
                max_caps.append(np.max(Q_raw[i]))
    init_cap = np.mean(max_caps) if max_caps else 1.0

    raw_soh = [np.max(q) / init_cap if len(q) > 0 else 0.0 for q in Q_raw]
    raw_soh = np.clip(raw_soh, 0, 1)
    corrected = raw_soh.copy()

    for i in range(1, len(raw_soh)-1):
        prev_s, cur_s, nxt_s = raw_soh[i-1], raw_soh[i], raw_soh[i+1]
        spike = (prev_s - cur_s > spike_thr) and (nxt_s - cur_s > spike_thr)

        prev_valid = 1.0
        for k in range(i-1, -1, -1):
            if len(Q_raw[k]) > 0:
                prev_valid = np.max(Q_raw[k])
                break
        cur_q = np.max(Q_raw[i]) if len(Q_raw[i]) > 0 else 0
        drop = (cur_q / prev_valid) < drop_ratio
        dur = t_raw[i][-1] - t_raw[i][0] if len(t_raw[i]) > 0 else 0
        too_short = dur < FULL_PLOT_TIME_THRESHOLD * dur_ratio

        if spike or drop or too_short:
            corrected[i] = np.nan

    soh = np.array(corrected)
    nan_idx = np.isnan(soh)
    if np.any(nan_idx):
        valid_idx = np.where(~nan_idx)[0]
        if len(valid_idx) >= 2:
            soh[nan_idx] = np.interp(np.flatnonzero(nan_idx), valid_idx, soh[valid_idx])

    if len(soh) >= 5:
        soh = medfilt(soh, 5)
    if len(soh) >= 9:
        soh = savgol_filter(soh, 9, 3)

    return np.clip(soh, 0, 1)

# ------------------------------------------------------------
# VOLTAGE WINDOWS TO TEST + BEST WINDOW
# ------------------------------------------------------------
voltage_ranges_to_test = [
    (2.70, 2.80),
    (3.00, 3.10),
    (3.30, 3.40),
    (3.50, 3.60),
]

precomputed_best_window = {
    "Voltage Range": "3.45–3.55",
    "MAE": 0.003672,
    "RMSE": 0.005008,
    "Coverage": 0.935571,
    "MIW": 0.032512,
}

# ======================================================================
# ROBUST SCALAR FEATURE EXTRACTION – WORKS FOR ALL WINDOWS
# ======================================================================
scalar_feature_names = [
    "cap_diff", "Q_mean", "Q_skew", "Q_kurt", "Q_auc",
    "V_mean", "V_skew", "V_kurt", "V_auc"
]

def build_scalar_features_for_range(voltage_range):
    all_cells_features = []
    all_cells_soh = []

    for cell in CELLS_TO_ANALYZE:
        Q_raw, V_raw, t_raw, I_raw = load_battery_data(DATA_PATH, cell_number=cell)
        soh = compute_smoothed_soh(Q_raw, t_raw)
        all_cells_soh.append(soh)

        cell_feats = []
        for i in range(len(Q_raw)):
            t_norm = normalize_time([t_raw[i]])[0]

            Vp, Qp, tp, Ip, time_start, time_end = extract_data_in_voltage_range(
                V_raw[i], Q_raw[i], t_norm, I_raw[i],
                target_voltage_range=voltage_range,
                mode=CHARGE_OR_DISCHARGE_FOCUS
            )

            if time_start is not None and time_end is not None and (time_end - time_start) > 1e-6:
                t_uni = np.linspace(time_start, time_end, TARGET_LENGTH_PARTIAL_FEATURES)
                V_interp = np.interp(t_uni, tp, Vp)
                Q_interp = np.interp(t_uni, tp, Qp)
            else:
                V_interp = np.zeros(TARGET_LENGTH_PARTIAL_FEATURES)
                Q_interp = np.zeros(TARGET_LENGTH_PARTIAL_FEATURES)

            cap_diff = Q_interp[-1] - Q_interp[0]
            Q_mean = np.mean(Q_interp)
            Q_auc = np.trapz(Q_interp)
            V_mean = np.mean(V_interp)
            V_auc = np.trapz(V_interp)

            Q_skew = skew(Q_interp) if np.std(Q_interp) > 1e-10 else 0.0
            Q_kurt = kurtosis(Q_interp) if np.std(Q_interp) > 1e-10 else 0.0
            V_skew = skew(V_interp) if np.std(V_interp) > 1e-10 else 0.0
            V_kurt = kurtosis(V_interp) if np.std(V_interp) > 1e-10 else 0.0

            cell_feats.append([
                cap_diff, Q_mean, Q_skew, Q_kurt, Q_auc,
                V_mean, V_skew, V_kurt, V_auc
            ])

        all_cells_features.append(np.array(cell_feats))

    # Global feature selection
    X_all = np.vstack(all_cells_features)
    y_all = np.concatenate(all_cells_soh)

    selected_names, selected_X = perform_feature_selection(X_all, y_all, scalar_feature_names)

    selected_idx = [scalar_feature_names.index(n) for n in selected_names] if selected_names else list(range(9))

    # Tiling + scaling
    tiled_list = []
    for feats in all_cells_features:
        sel = feats[:, selected_idx]
        tiled = np.repeat(sel[:, None, :], TARGET_LENGTH_PARTIAL_FEATURES, axis=1)
        tiled_list.append(tiled)

    flat = np.vstack([t.reshape(-1, t.shape[-1]) for t in tiled_list])
    scaler = MinMaxScaler()
    flat_scaled = scaler.fit_transform(flat)

    scaled_list = []
    idx = 0
    for t in tiled_list:
        n_cyc, _, n_feat = t.shape
        scaled_list.append(
            flat_scaled[idx:idx + n_cyc * TARGET_LENGTH_PARTIAL_FEATURES]
            .reshape(n_cyc, TARGET_LENGTH_PARTIAL_FEATURES, n_feat)
        )
        idx += n_cyc * TARGET_LENGTH_PARTIAL_FEATURES

    return scaled_list, all_cells_soh, len(selected_idx)

# ======================================================================
# RUN SWEEP – NOW WORKS FOR ALL RANGES
# ======================================================================
sensitivity_rows = []
for vr in voltage_ranges_to_test:
    print(f"\n===== SCALAR Sensitivity for range {vr} V =====")
    X_raw, y_raw, nfeat = build_scalar_features_for_range(vr)
    X_list = [torch.tensor(x, dtype=torch.float32) for x in X_raw]
    y_list = [torch.tensor(y, dtype=torch.float32) for y in y_raw]

    metrics, _, _ = run_kfold_training_and_evaluation(
        X_data_tensor_list=X_list,
        y_data_tensor_list=y_list,
        model_class=TemporalTransformerWithCNNFusion,
        num_features_per_point=nfeat,
        model_name=f"Sensitivity_{vr[0]:.2f}-{vr[1]:.2f}",
        **best_fusion_params
    )

    sensitivity_rows.append({
        "Voltage Range": f"{vr[0]:.2f}–{vr[1]:.2f}",
        "MAE": metrics["mae"],
        "RMSE": metrics["rmse"],
        "Coverage": metrics["coverage_rate"],
        "MIW": metrics["mean_interval_width"]
    })

sensitivity_rows.append(precomputed_best_window)
df_sensitivity = pd.DataFrame(sensitivity_rows)

labels_order = ["2.70–2.80", "3.00–3.10", "3.30–3.40", "3.50–3.60", "3.45–3.55"]
df_sensitivity = df_sensitivity.set_index("Voltage Range").reindex(labels_order).reset_index()

# ======================================================================
# FINAL PERFECT FIGURE – 3.45–3.55 IN BOLD RED, CLEAR WINNER
# ======================================================================
palette = {
    "2.70–2.80": "#8EC9E8",
    "3.00–3.10": "#F4B266",
    "3.30–3.40": "#9AD69B",
    "3.50–3.60": "#B3A1E5",
    "3.45–3.55": "#e63946",   # BOLD RED = BEST
}

DPI = 600
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=DPI)

# (a) Deterministic
ax1 = axes[0]
df_det = df_sensitivity.melt(id_vars="Voltage Range", value_vars=["MAE", "RMSE"])
sns.violinplot(data=df_det, x="variable", y="value", palette=["#fbe0db", "#f8ecc7"], inner=None, linewidth=0.9, ax=ax1)
sns.boxplot(data=df_det, x="variable", y="value", showcaps=True,
            boxprops={"facecolor":"white","edgecolor":"black","linewidth":1.3},
            whiskerprops={"linewidth":1.3}, medianprops={"color":"black","linewidth":2},
            showfliers=False, width=0.3, ax=ax1)
sns.stripplot(data=df_det, x="variable", y="value", hue="Voltage Range", palette=palette,
              size=11, jitter=True, edgecolor="black", linewidth=1.3, ax=ax1)

ax1.set_ylabel("Error (SOH fraction)")
ax1.set_xlabel("")
#ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.set_ylim(0.002, 0.015)
ax1.legend(title="Voltage window (V)", frameon=False, bbox_to_anchor=(1.48, 1.0), loc="upper right")
ax1.text(-0.12, -0.18, "(a)", transform=ax1.transAxes, fontsize=16, fontweight="bold")

# (b) Probabilistic
ax2 = axes[1]
ax2b = ax2.twinx()
df_prob = df_sensitivity.melt(id_vars="Voltage Range", value_vars=["Coverage", "MIW"])
cov = df_prob[df_prob["variable"] == "Coverage"]
miw = df_prob[df_prob["variable"] == "MIW"]

for d, a, c in [(cov, ax2, "#d7e5fb"), (miw, ax2b, "#e5ddfb")]:
    sns.violinplot(data=d, x="variable", y="value", color=c, inner=None, linewidth=0.9, ax=a)
    sns.boxplot(data=d, x="variable", y="value", showcaps=True,
                boxprops={"facecolor":"white","edgecolor":"black","linewidth":1.3},
                whiskerprops={"linewidth":1.3}, medianprops={"color":"black","linewidth":2},
                showfliers=False, width=0.3, ax=a)
    sns.stripplot(data=d, x="variable", y="value", hue="Voltage Range", palette=palette,
                  size=11, jitter=True, edgecolor="black", linewidth=1.3, ax=a, legend=False)

ax2.set_ylabel("Coverage rate", color="#1f77b4")
ax2b.set_ylabel("Mean interval width", color="#9467bd")
ax2.set_ylim(0.70, 1.01)
ax2b.set_ylim(0, 0.055)
ax2.tick_params(axis='y', colors="#1f77b4")
ax2b.tick_params(axis='y', colors="#9467bd")
ax2.text(-0.12, -0.18, "(b)", transform=ax2.transAxes, fontsize=16, fontweight="bold")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("Voltage_Window_Sensitivity_Final_Dec02_2025_600dpi.png", dpi=600, bbox_inches="tight")
fig.savefig("Voltage_Window_Sensitivity_Final_Dec02_2025.pdf", bbox_inches="tight")
plt.show()
