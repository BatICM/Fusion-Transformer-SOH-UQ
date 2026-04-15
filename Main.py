import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pywt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from scipy.signal import savgol_filter, medfilt
from itertools import product
import collections
from scipy.stats import skew, kurtosis
import numpy.typing as npt
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler


def set_q1_style(
    dpi: int = 600,
    base_fontsize: float = 9.0,
    use_tex: bool = False,
):
   
    q1_colors = [
        "#1f77b4",  # deep blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#17becf",  # teal
        "#8c564b",  # brown
        "#7f7f7f",  # gray
    ]

    mpl.rcParams["figure.dpi"] = dpi
    mpl.rcParams["savefig.dpi"] = dpi
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0.03

    mpl.rcParams["text.usetex"] = use_tex
    mpl.rcParams["font.family"] = "DejaVu Sans" if not use_tex else "serif"
    mpl.rcParams["font.size"] = base_fontsize
    mpl.rcParams["axes.titlesize"] = base_fontsize + 3
    mpl.rcParams["axes.labelsize"] = base_fontsize + 2
    mpl.rcParams["legend.fontsize"] = base_fontsize - 1
    mpl.rcParams["xtick.labelsize"] = base_fontsize - 1
    mpl.rcParams["ytick.labelsize"] = base_fontsize - 1

    # --- Axes & spines ---
    mpl.rcParams["axes.linewidth"] = 1.1
    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["axes.grid"] = False        # you add grids manually
    mpl.rcParams["axes.facecolor"] = "white"
    mpl.rcParams["axes.prop_cycle"] = cycler(color=q1_colors)

    # --- Lines & markers ---
    mpl.rcParams["lines.linewidth"] = 1.8
    mpl.rcParams["lines.markersize"] = 5
    mpl.rcParams["lines.markeredgewidth"] = 0.5

    # --- Ticks ---
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.major.size"] = 4
    mpl.rcParams["ytick.major.size"] = 4
    mpl.rcParams["xtick.minor.size"] = 2
    mpl.rcParams["ytick.minor.size"] = 2
    mpl.rcParams["xtick.major.width"] = 0.8
    mpl.rcParams["ytick.major.width"] = 0.8
    mpl.rcParams["xtick.minor.width"] = 0.6
    mpl.rcParams["ytick.minor.width"] = 0.6

    # --- Legends ---
    mpl.rcParams["legend.frameon"] = True
    mpl.rcParams["legend.framealpha"] = 0.9
    mpl.rcParams["legend.facecolor"] = "white"
    mpl.rcParams["legend.edgecolor"] = "0.3"
    mpl.rcParams["legend.fancybox"] = True

    # --- Colormap defaults (for imshow/contourf etc.) ---
    mpl.rcParams["image.cmap"] = "viridis"

    # --- Figure layout ---
    mpl.rcParams["figure.figsize"] = (6.5, 4.0)  # default size; you can override per-figure
    mpl.rcParams["figure.autolayout"] = False    # you are using tight_layout() / constrained_layout()

    # --- Patch / bar defaults ---
    mpl.rcParams["patch.edgecolor"] = "black"
    mpl.rcParams["patch.linewidth"] = 1.0

    print(f"[Q1 style] Applied (dpi={dpi}, base_fontsize={base_fontsize}, use_tex={use_tex})")


np.random.seed(42)
torch.manual_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


DATA_PATH = 'C:/Users/machenike/Desktop/matlab_data1_1_10.mat'

CELLS_TO_ANALYZE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
CELL_TO_ANALYZE_PLOT_EXAMPLE = 9
CYCLE_TO_ANALYZE_PLOT_EXAMPLE = 1
Q_DIFF_EXAMPLE_CYCLE = 800
CHARGE_OR_DISCHARGE_FOCUS = 'charge'
FULL_OPERATING_VOLTAGE_RANGE = (2.6, 3.6)
FULL_PLOT_TIME_THRESHOLD = 60.0

FEATURE_EXTRACTION_VOLTAGE_RANGE = (3.45, 3.55)
TARGET_LENGTH_PARTIAL_FEATURES = 100
DPI = 600 
FIGSIZE_SUBPLOT = (4, 3) 

FEATURES_TO_INCLUDE = {
    'voltage': False,             
    'capacity': False,             
    'wavelet_capacity': False,     
    'dq_dv': False,                
    'dv_dq': False,                
    'ir_profile': False,           
    'capacity_difference': True,   
    'q_profile_stats': True,       
    'v_profile_stats': True,       
}
# ============================================================================

# --- SOH Correction Parameters ---
SOH_CORRECTION_PARAMS = {
    'initial_capacity_avg_cycles': 5,
    'capacity_drop_ratio_threshold': 0.80,
    'soh_spike_threshold': 0.05,
    'min_cycle_duration_ratio': 0.2
}

def set_q1_style():
    plt.rcParams.update({
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "font.family": "Arial",
        "axes.linewidth": 1.0,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": 9,
        "legend.frameon": True,
    })

set_q1_style()

def moving_average(data, window_size=5):
    """
    Apply moving average smoothing to data.
    """
    if len(data) < window_size:
        return data
    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    pad_front = (window_size - 1) // 2
    pad_back = window_size - 1 - pad_front

    if smoothed.size > 0:
        return np.pad(smoothed, (pad_front, pad_back), mode='edge')
    else:
        return data

# Function to calculate AUC using trapezoidal rule
def calculate_auc(y_values):
    if len(y_values) < 2:
        return 0.0
    return np.trapz(y_values, dx=1.0)


def extract_wavelet_features(data, wavelet='db4', level=3, target_length=None):
    """
    Extract wavelet coefficients from time-series data at multiple resolution levels.
    """
    if data.shape[0] == 0:
        return np.array([])
    if data.ndim == 1:
        data = data[np.newaxis, :]

    if target_length is None:
        target_length = data.shape[1]

    wavelet_coeffs = []
    for sample_idx, sample in enumerate(data):
        if len(sample) < 2:
            wavelet_coeffs.append(np.zeros(target_length))
            continue
        try:
            coeffs = pywt.wavedec(sample, wavelet, level=level, mode='periodic')
            concat_coeffs = np.concatenate([coeffs[0], *coeffs[1:]])
            padded_coeffs = np.pad(concat_coeffs, (0, max(0, target_length - len(concat_coeffs))),
                                  mode='edge')[:target_length]
            wavelet_coeffs.append(padded_coeffs)
        except Exception as e:
            print(f"Error during wavelet transform for sample {sample_idx}: {e}. Filling with zeros.")
            wavelet_coeffs.append(np.zeros(target_length))
    if not wavelet_coeffs:
        return np.array([])

    return np.array(wavelet_coeffs)

def load_battery_data(file_path, cell_number=0, max_cycles=None):
    
    print(f"Loading data from: {file_path}")
    data = loadmat(file_path)['batch1']

    total_cycles_available = len(data[cell_number][2][0])
    if max_cycles is None or max_cycles > total_cycles_available:
        cycles_to_load = data[cell_number][2][0][:total_cycles_available]
        print(f"Loading all {total_cycles_available} cycles for cell {cell_number}.")
    else:
        cycles_to_load = data[cell_number][2][0][:max_cycles]
        print(f"Loading {max_cycles} cycles for cell {cell_number}.")

    Q, V, t, I = [], [], [], []
    for cycle in cycles_to_load:
        Q.append(np.array(cycle[2]).flatten())
        V.append(np.array(cycle[4]).flatten())
        t.append(np.array(cycle[1]).flatten())
        I.append(np.array(cycle[3]).flatten())

    return Q, V, t, I

def normalize_time(time_data_list):
    
    normalized_times = []
    for t_cycle in time_data_list:
        if len(t_cycle) > 0:
            normalized_times.append(t_cycle - t_cycle[0])
        else:
            normalized_times.append(np.array([]))
    return normalized_times

def filter_by_time_and_voltage(Q_list, V_list, t_list, I_list, voltage_range=None, time_threshold=None, mode='both'):
    
    filtered_Q, filtered_V, filtered_t, filtered_I = [], [], [], []
    for q, v, t_cycle, i in zip(Q_list, V_list, t_list, I_list):
        if len(t_cycle) == 0:
            filtered_Q.append(np.array([]))
            filtered_V.append(np.array([]))
            filtered_t.append(np.array([]))
            filtered_I.append(np.array([]))
            continue

        mask = np.full_like(t_cycle, True, dtype=bool)

        if voltage_range is not None:
            min_V, max_V = voltage_range
            voltage_mask = (v >= min(min_V, max_V)) & (v <= max(min_V, max(max_V, v.max())))
            mask = mask & voltage_mask

        if time_threshold is not None:
            time_mask = t_cycle < time_threshold
            mask = mask & time_mask

        if mode == 'charge':
            current_mask = (i > 0.001)
            mask = mask & current_mask
        elif mode == 'discharge':
            current_mask = (i < -0.001)
            mask = mask & current_mask
        elif mode != 'both':
            raise ValueError("Mode must be 'charge', 'discharge', or 'both'.")

        filtered_Q.append(q[mask])
        filtered_V.append(v[mask])
        filtered_t.append(t_cycle[mask])
        filtered_I.append(i[mask])
    return filtered_Q, filtered_V, filtered_t, filtered_I


def interpolate_data(t_list, V_list, Q_list, I_list, num_points=100, interpolation_time_range=None):
    
    V_interp, Q_interp, I_interp = [], [], []
    uniform_t_generated = None

    for cycle_idx, (t_cycle, v, q, i) in enumerate(zip(t_list, V_list, Q_list, I_list)):
        if len(t_cycle) < 2:
            V_interp.append(np.zeros(num_points))
            Q_interp.append(np.zeros(num_points))
            I_interp.append(np.zeros(num_points))
            continue

        sort_indices = np.argsort(t_cycle)
        t_cycle_sorted = t_cycle[sort_indices]
        v_sorted = v[sort_indices]
        q_sorted = q[sort_indices]
        i_sorted = i[sort_indices]

        _, unique_indices = np.unique(t_cycle_sorted, return_index=True)
        if len(unique_indices) != len(t_cycle_sorted):
            t_cycle_sorted = t_cycle_sorted[unique_indices]
            v_sorted = v_sorted[unique_indices]
            q_sorted = q_sorted[unique_indices]
            i_sorted = i_sorted[unique_indices]

        if t_cycle_sorted.shape[0] < 2:
            V_interp.append(np.zeros(num_points))
            Q_interp.append(np.zeros(num_points))
            I_interp.append(np.zeros(num_points))
            continue

        if interpolation_time_range is not None:
            time_start, time_end = interpolation_time_range
        else:
            time_start, time_end = t_cycle_sorted[0], t_cycle_sorted[-1]

        if time_end - time_start < 1e-6:
            uniform_t_current = np.full(num_points, time_start)
        else:
            uniform_t_current = np.linspace(time_start, time_end, num_points)

        V_interp.append(np.interp(uniform_t_current, t_cycle_sorted, v_sorted))
        Q_interp.append(np.interp(uniform_t_current, t_cycle_sorted, q_sorted))
        I_interp.append(np.interp(uniform_t_current, t_cycle_sorted, i_sorted))

        if uniform_t_generated is None:
            uniform_t_generated = uniform_t_current

    if not V_interp:
        uniform_t_generated = np.linspace(0, 1, num_points)
        return uniform_t_generated, np.array([]), np.array([]), np.array([])

    return uniform_t_generated, np.array(V_interp), np.array(Q_interp), np.array(I_interp)


def extract_data_in_voltage_range(V_cycle, Q_cycle, t_cycle, I_cycle, target_voltage_range, mode='both'):
    
    min_V, max_V = target_voltage_range

    voltage_mask = (V_cycle >= min(min_V, max_V)) & (V_cycle <= max(min_V, max(max_V, V_cycle.max())))

    if mode == 'charge':
        current_mask = (I_cycle > 0.001)
    elif mode == 'discharge':
        current_mask = (I_cycle < -0.001)
    elif mode == 'both':
        current_mask = np.full_like(I_cycle, True, dtype=bool)
    else:
        raise ValueError("Mode must be 'charge', 'discharge', or 'both'.")

    combined_mask = voltage_mask & current_mask

    if not np.any(combined_mask):
        return np.array([]), np.array([]), np.array([]), np.array([]), None, None

    V_partial = V_cycle[combined_mask]
    Q_partial = Q_cycle[combined_mask]
    t_partial = t_cycle[combined_mask]
    I_partial = I_cycle[combined_mask]

    if len(t_partial) > 0:
        time_start = t_partial[0]
        time_end = t_partial[-1]
    else:
        time_start = None
        time_end = None

    sort_by_time_indices = np.argsort(t_partial)
    V_partial = V_partial[sort_by_time_indices]
    Q_partial = Q_partial[sort_by_time_indices]
    t_partial = t_partial[sort_by_time_indices]
    I_partial = I_partial[sort_by_time_indices]

    _, unique_time_indices = np.unique(t_partial, return_index=True)
    if len(unique_time_indices) != len(t_partial):
        t_partial = t_partial[unique_time_indices]
        V_partial = V_partial[unique_time_indices]
        Q_partial = Q_partial[unique_time_indices]
        I_partial = I_partial[unique_time_indices]

    return V_partial, Q_partial, t_partial, I_partial, time_start, time_end


# Function to perform feature selection as per the defined procedure ---
def perform_feature_selection(features: npt.NDArray[np.float64],
                              SOH: npt.NDArray[np.float64],
                              feature_names: list,
                              rho_threshold=0.5,
                              rho_ij_threshold=0.9):
    
    if features.shape[0] == 0 or SOH.shape[0] == 0:
        return [], np.array([])

    min_len = min(features.shape[0], SOH.shape[0])
    features = features[:min_len]
    SOH = SOH[:min_len]

    # Step 1-2: Calculate and average correlation with SOH
    rho_i = np.array([np.corrcoef(features[:, i], SOH)[0, 1] for i in range(features.shape[1])])
    rho_i_abs = np.abs(rho_i)

    # Step 3: Filter and rank features
    filtered_indices = np.where(rho_i_abs >= rho_threshold)[0]
    filtered_feature_names = [feature_names[i] for i in filtered_indices]
    filtered_features = features[:, filtered_indices]
    filtered_rho_i_abs = rho_i_abs[filtered_indices]

    # Rank from large to small
    sorted_indices = np.argsort(filtered_rho_i_abs)[::-1]
    F_indices = filtered_indices[sorted_indices].tolist()
    F_names = [feature_names[i] for i in F_indices]

    # Step 5: Remove redundant features
    i = 0
    while i < len(F_indices):
        j = i + 1
        while j < len(F_indices):
            f_i_idx = F_indices[i]
            f_j_idx = F_indices[j]
            rho_ij = np.corrcoef(features[:, f_i_idx], features[:, f_j_idx])[0, 1]
            if np.abs(rho_ij) > rho_ij_threshold:
                print(f"  - Deleting redundant feature: '{F_names[j]}' (correlated with '{F_names[i]}', rho={rho_ij:.4f})")
                F_indices.pop(j)
                F_names.pop(j)
            else:
                j += 1
        i += 1

    selected_features = features[:, F_indices]
    return F_names, selected_features

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set global font to Times New Roman
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['savefig.dpi'] = 600

def plot_overview_and_sequence(ax1, ax2, V_full_interp, I_full_interp, t_full_interp,
                               time_highlight_start, time_highlight_end,
                               V_partial_features, Q_partial_features,
                               full_voltage_range, feature_extraction_voltage_range,
                               cycle_num, cell_num, mode, target_length, features_to_plot):
    title_fontsize = 13
    label_fontsize = 11
    tick_fontsize = 9
    line_width = 1.8
    # ============================
    # LEFT SUBPLOT (full cycle)
    # ============================
    ax1.set_facecolor("white")
    if V_full_interp is not None and V_full_interp.size > 0:
        # Voltage vs time (primary axis)
        ax1.plot(
            t_full_interp,
            V_full_interp[0],
            color="#1f77b4",
            linewidth=line_width,
            solid_capstyle="round",
            label="Voltage"
        )
        ax1.set_xlabel("Time (min)", fontsize=label_fontsize, fontname='Times New Roman')
        ax1.set_ylabel("Voltage (V)", color="#1f77b4", fontsize=label_fontsize, fontname='Times New Roman')
        ax1.tick_params(axis="y", labelcolor="#1f77b4", labelsize=tick_fontsize)
        ax1.tick_params(axis="x", labelsize=tick_fontsize)
        # Current vs time (secondary axis)
        ax1b = ax1.twinx()
        ax1b.plot(
            t_full_interp,
            I_full_interp[0],
            color="#d62728",
            linestyle="--",
            linewidth=line_width,
            solid_capstyle="round",
            label="Current"
        )
        ax1b.set_ylabel("Current (A)", color="#d62728", fontsize=label_fontsize, fontname='Times New Roman')
        ax1b.tick_params(axis="y", labelcolor="#d62728", labelsize=tick_fontsize)
        # Highlight the full operating voltage range time window if available
        if time_highlight_start is not None and time_highlight_end is not None:
            ax1.axvspan(
                time_highlight_start,
                time_highlight_end,
                color="lightgray",
                alpha=0.3,
                zorder=0
            )
        ax1.set_xlim(t_full_interp[0], t_full_interp[-1])
        ax1.set_ylim(min(full_voltage_range) - 0.05, max(full_voltage_range) + 0.05)
    else:
        ax1.set_xlabel("Time (min)", fontsize=label_fontsize, fontname='Times New Roman')
        ax1.set_ylabel("Voltage (V)", fontsize=label_fontsize, fontname='Times New Roman')
    # Add (a) label below bottom-left, outside the axis
    ax1.text(-0.12, -0.18, "(a)", transform=ax1.transAxes, fontsize=12, fontweight="bold")
    for spine in ax1.spines.values():
        spine.set_linewidth(1.0)
    # ============================
    # RIGHT SUBPLOT (features)
    # ============================
    ax2.set_facecolor("white")
    if V_partial_features is not None and V_partial_features.size > 0:
        x = np.arange(target_length)
        # Voltage sequence (primary y)
        ax2.plot(
            x,
            V_partial_features,
            color="#005f73",
            marker="o",
            markersize=3.5,
            linewidth=line_width + 0.2,
            solid_capstyle="round",
            label="Voltage sequence"
        )
        ax2.set_xlabel(f"Interpolated Index (Length {target_length})", fontsize=label_fontsize, fontname='Times New Roman')
        ax2.set_ylabel("Voltage (V)", color="#005f73", fontsize=label_fontsize, fontname='Times New Roman')
        ax2.tick_params(axis="y", labelcolor="#005f73", labelsize=tick_fontsize)
        ax2.tick_params(axis="x", labelsize=tick_fontsize)
        # Capacity sequence (secondary y) – single solid color
        ax2b = ax2.twinx()
        capacity_color = "#e67e22"
        ax2b.plot(
            x,
            Q_partial_features,
            color=capacity_color,
            linestyle="--",
            linewidth=line_width + 0.2,
            solid_capstyle="round"
        )
        ax2b.set_ylabel("Capacity (Ah)", color=capacity_color, fontsize=label_fontsize, fontname='Times New Roman')
        ax2b.tick_params(axis="y", labelcolor=capacity_color, labelsize=tick_fontsize)
        # Highlight feature extraction voltage window as band in voltage axis
        band = ax2.axhspan(
            feature_extraction_voltage_range[0],
            feature_extraction_voltage_range[1],
            facecolor="#f5f5b5",
            alpha=0.5,
            zorder=0
        )
        # Legend: voltage + feature window
        voltage_proxy = Line2D(
            [0], [0],
            color="#005f73",
            marker="o",
            linewidth=line_width + 0.2,
            label="Voltage sequence"
        )
        band_proxy = Line2D(
            [0], [0],
            color="#f5f5b5",
            linewidth=8,
            label=f"Feature window {feature_extraction_voltage_range[0]:.2f}–{feature_extraction_voltage_range[1]:.2f} V"
        )
        ax2.legend(
            handles=[voltage_proxy, band_proxy],
            loc="lower right",
            frameon=True,
            facecolor="white",
            edgecolor="lightgray",
            prop={'family': 'Times New Roman', 'size': tick_fontsize}
        )
        
        ax2.set_xlim(x.min(), x.max())
        ax2.set_ylim(
            min(V_partial_features.min(), feature_extraction_voltage_range[0]) - 0.005,
            max(V_partial_features.max(), feature_extraction_voltage_range[1]) + 0.005
        )
        ax2b.set_ylim(Q_partial_features.min() - 0.02, Q_partial_features.max() + 0.02)
    else:
        ax2.set_xlabel(f"Interpolated Index (Length {target_length})", fontsize=label_fontsize, fontname='Times New Roman')
        ax2.set_ylabel("Voltage (V)", fontsize=label_fontsize, fontname='Times New Roman')
    # Add (b) label below bottom-left, outside the axis
    ax2.text(-0.12, -0.18, "(b)", transform=ax2.transAxes, fontsize=12, fontweight="bold")
    for spine in ax2.spines.values():
        spine.set_linewidth(1.0)
    plt.tight_layout()
    fig.savefig('figure.png', dpi=600, bbox_inches='tight')
    plt.show()
print("Figure saved at 600 DPI.")
        
        
        
# --- Main Data Processing Loop (Step 1) ---
all_cells_features_list = []
all_cells_soh_list = []
all_cells_soh_lists_per_cell = []
all_cells_all_features_flattened = []

plot_example_V_full_raw_interp = None
plot_example_Q_partial_features = None
plot_example_I_full_raw_interp = None
plot_example_t_full_raw_interp = None
plot_example_time_highlight_start_full_range = None
plot_example_time_highlight_end_full_range = None
plot_example_cell_num = None
plot_example_cycle_num = None
plot_example_V_partial_features = None
plot_example_dQdV_partial_features = None

initial_partial_capacities_per_cell = {}


for cell_idx, cell_num in enumerate(CELLS_TO_ANALYZE):
    print(f"\n--- Processing data for Cell: {cell_num} ---")

    Q_raw, V_raw, t_raw, I_raw = load_battery_data(DATA_PATH, cell_number=cell_num)

    initial_cycles_for_capacity = SOH_CORRECTION_PARAMS['initial_capacity_avg_cycles']
    max_capacities_initial_cycles = []
    for i in range(min(initial_cycles_for_capacity, len(Q_raw))):
        if len(Q_raw[i]) > 0 and t_raw[i].size > 0 and (t_raw[i][-1] - t_raw[i][0]) > (FULL_PLOT_TIME_THRESHOLD * SOH_CORRECTION_PARAMS['min_cycle_duration_ratio']):
            max_capacities_initial_cycles.append(np.max(Q_raw[i]))

    if max_capacities_initial_cycles:
        initial_capacity_current_cell = np.mean(max_capacities_initial_cycles)
    else:
        print(f"Warning: Cell {cell_num} has no sufficient *valid* data for initial capacity. Setting to 1.0 for SOH calculation.")
        initial_capacity_current_cell = 1.0

    raw_soh_current_cell = []
    for q_cycle in Q_raw:
        if initial_capacity_current_cell > 0:
            soh_val = np.max(q_cycle) / initial_capacity_current_cell if len(q_cycle) > 0 else 0.0
            raw_soh_current_cell.append(soh_val)
        else:
            raw_soh_current_cell.append(0.0)

    raw_soh_current_cell = np.clip(raw_soh_current_cell, 0.0, 1.0)

    corrected_soh_current_cell = list(raw_soh_current_cell)
    valid_soh_indices = [i for i, soh in enumerate(raw_soh_current_cell) if i < 2 or (raw_soh_current_cell[i] > raw_soh_current_cell[i-1] - SOH_CORRECTION_PARAMS['soh_spike_threshold'] * 2)]

    for i in range(1, len(raw_soh_current_cell) - 1):
        current_soh = raw_soh_current_cell[i]
        prev_soh = raw_soh_current_cell[i-1]
        next_soh = raw_soh_current_cell[i+1]

        is_soh_spike = (prev_soh - current_soh > SOH_CORRECTION_PARAMS['soh_spike_threshold']) and \
                       (next_soh - current_soh > SOH_CORRECTION_PARAMS['soh_spike_threshold']) and \
                       (next_soh - prev_soh < SOH_CORRECTION_PARAMS['soh_spike_threshold'] / 2)

        prev_valid_max_q = 0.0
        for k in range(i-1, -1, -1):
            if k in valid_soh_indices:
                if len(Q_raw[k]) > 0:
                    prev_valid_max_q = np.max(Q_raw[k])
                break

        current_max_q = np.max(Q_raw[i]) if len(Q_raw[i]) > 0 else 0.0
        is_capacity_drop = False
        if prev_valid_max_q > 0.001:
             if (current_max_q / prev_valid_max_q) < SOH_CORRECTION_PARAMS['capacity_drop_ratio_threshold']:
                 is_capacity_drop = True

        current_cycle_duration = t_raw[i][-1] - t_raw[i][0] if len(t_raw[i]) > 0 else 0
        is_short_cycle = current_cycle_duration < (FULL_PLOT_TIME_THRESHOLD * SOH_CORRECTION_PARAMS['min_cycle_duration_ratio'])

        if is_soh_spike or is_capacity_drop or is_short_cycle:
            corrected_soh_current_cell[i] = np.nan

    soh_array = np.array(corrected_soh_current_cell)
    nan_indices = np.where(np.isnan(soh_array))[0]
    if len(nan_indices) > 0:
        non_nan_indices = np.where(~np.isnan(soh_array))[0]
        if len(non_nan_indices) >= 2:
            soh_array[nan_indices] = np.interp(nan_indices, non_nan_indices, soh_array[non_nan_indices])

    median_filter_window = 5
    if len(soh_array) >= median_filter_window:
        soh_median_filtered = medfilt(soh_array, kernel_size=median_filter_window)
    else:
        soh_median_filtered = soh_array

    savgol_window = 9
    savgol_poly = 3
    if len(soh_median_filtered) >= savgol_window and savgol_window > savgol_poly:
        SOH_smoothed_current_cell = savgol_filter(soh_median_filtered, window_length=savgol_window, polyorder=savgol_poly)
    else:
        SOH_smoothed_current_cell = soh_median_filtered

    SOH_smoothed_current_cell = np.clip(SOH_smoothed_current_cell, 0.0, 1.0)
    all_cells_soh_lists_per_cell.append(SOH_smoothed_current_cell)
    all_cells_soh_list.append(SOH_smoothed_current_cell)

    current_cell_features = []
    current_cell_all_features_flattened = []

    for i in range(len(V_raw)):
        current_t_normalized = normalize_time([t_raw[i]])[0]

        Q_full_range_filtered, V_full_range_filtered, t_full_range_filtered, I_full_range_filtered = \
            filter_by_time_and_voltage([Q_raw[i]], [V_raw[i]], [current_t_normalized], [I_raw[i]],
                                       voltage_range=FULL_OPERATING_VOLTAGE_RANGE,
                                       time_threshold=FULL_PLOT_TIME_THRESHOLD,
                                       mode=CHARGE_OR_DISCHARGE_FOCUS)

        uniform_t_full_interp_i, V_interp_full_i, Q_interp_full_i, I_interp_full_i = interpolate_data(
            t_full_range_filtered, V_full_range_filtered, Q_full_range_filtered, I_full_range_filtered,
            num_points=100,
            interpolation_time_range=None
        )

        time_highlight_start_full_range = uniform_t_full_interp_i[0] if uniform_t_full_interp_i.size > 0 else None
        time_highlight_end_full_range = uniform_t_full_interp_i[-1] if uniform_t_full_interp_i.size > 0 else None


        V_partial_raw, Q_partial_raw, t_partial_raw, I_partial_raw, _, _ = \
            extract_data_in_voltage_range(V_raw[i], Q_raw[i],
                                          current_t_normalized, I_raw[i],
                                          FEATURE_EXTRACTION_VOLTAGE_RANGE, CHARGE_OR_DISCHARGE_FOCUS)

        V_interp_p_cycle = np.zeros(TARGET_LENGTH_PARTIAL_FEATURES)
        Q_interp_p_cycle = np.zeros(TARGET_LENGTH_PARTIAL_FEATURES)
        Q_wavelet_p_cycle = np.zeros(TARGET_LENGTH_PARTIAL_FEATURES)
        DQDV_interp_p_cycle = np.zeros(TARGET_LENGTH_PARTIAL_FEATURES)
        DVDQ_interp_p_cycle = np.zeros(TARGET_LENGTH_PARTIAL_FEATURES)
        IR_profile_p_cycle = np.zeros(TARGET_LENGTH_PARTIAL_FEATURES)

        capacity_difference_feature = 0.0
        q_profile_mean, q_profile_skew, q_profile_kurt, q_profile_auc = 0.0, 0.0, 0.0, 0.0
        v_profile_mean, v_profile_skew, v_profile_kurt, v_profile_auc = 0.0, 0.0, 0.0, 0.0

        if t_partial_raw.size > 0:
            if len(t_partial_raw) >= 2 and (t_partial_raw[-1] - t_partial_raw[0]) > 0.001:
                uniform_t_for_features = np.linspace(t_partial_raw[0], t_partial_raw[-1], TARGET_LENGTH_PARTIAL_FEATURES)
                V_interp_p_cycle = np.interp(uniform_t_for_features, t_partial_raw, V_partial_raw)
                Q_interp_p_cycle = np.interp(uniform_t_for_features, t_partial_raw, Q_partial_raw)

                # Calculate derivatives (needed for scalar extraction even though profiles won't be used)
                v_smooth = savgol_filter(V_partial_raw, window_length=min(9, len(V_partial_raw)), polyorder=min(3, max(1, len(V_partial_raw)-1))) if len(V_partial_raw) > 5 else V_partial_raw
                q_smooth = savgol_filter(Q_partial_raw, window_length=min(9, len(Q_partial_raw)), polyorder=min(3, max(1, len(Q_partial_raw)-1))) if len(Q_partial_raw) > 5 else Q_partial_raw

                dq = np.gradient(q_smooth)
                dv = np.gradient(v_smooth)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    dq_dv_raw = np.divide(dq, dv, where=dv!=0)
                    dq_dv_raw[np.isinf(dq_dv_raw)] = 0.0
                    dq_dv_raw[np.isnan(dq_dv_raw)] = 0.0
                    
                    dv_dq_raw = np.divide(dv, dq, where=dq!=0)
                    dv_dq_raw[np.isinf(dv_dq_raw)] = 0.0
                    dv_dq_raw[np.isnan(dv_dq_raw)] = 0.0

                if len(t_partial_raw) > 1:
                    DQDV_interp_p_cycle = np.interp(uniform_t_for_features, t_partial_raw, dq_dv_raw)
                    DVDQ_interp_p_cycle = np.interp(uniform_t_for_features, t_partial_raw, dv_dq_raw)
                    
                    ir_raw = np.divide(V_partial_raw, I_partial_raw, where=I_partial_raw!=0)
                    ir_raw[np.isinf(ir_raw)] = 0.0
                    ir_raw[np.isnan(ir_raw)] = 0.0
                    IR_profile_p_cycle = np.interp(uniform_t_for_features, t_partial_raw, ir_raw)
            elif t_partial_raw.size == 1:
                V_interp_p_cycle = np.full(TARGET_LENGTH_PARTIAL_FEATURES, V_partial_raw[0])
                Q_interp_p_cycle = np.full(TARGET_LENGTH_PARTIAL_FEATURES, Q_partial_raw[0])
            else:
                pass

        if FEATURES_TO_INCLUDE['wavelet_capacity']:
            wavelet_result = extract_wavelet_features(np.array([Q_interp_p_cycle]), target_length=TARGET_LENGTH_PARTIAL_FEATURES)
            if wavelet_result.size > 0:
                Q_wavelet_p_cycle = wavelet_result[0]
            else:
                pass

        if FEATURES_TO_INCLUDE['capacity_difference']:
            if cell_num not in initial_partial_capacities_per_cell:
                initial_partial_capacities_per_cell[cell_num] = []
            if np.any(Q_interp_p_cycle != 0) and i < SOH_CORRECTION_PARAMS['initial_capacity_avg_cycles']:
                 initial_partial_capacities_per_cell[cell_num].append(Q_interp_p_cycle)

            avg_initial_partial_Q_for_diff = np.zeros(TARGET_LENGTH_PARTIAL_FEATURES)
            if cell_num in initial_partial_capacities_per_cell and len(initial_partial_capacities_per_cell[cell_num]) > 0:
                num_initial_samples_to_average = min(SOH_CORRECTION_PARAMS['initial_capacity_avg_cycles'], len(initial_partial_capacities_per_cell[cell_num]))
                avg_initial_partial_Q_for_diff = np.mean(initial_partial_capacities_per_cell[cell_num][:num_initial_samples_to_average], axis=0)

            if np.any(avg_initial_partial_Q_for_diff != 0) and np.any(Q_interp_p_cycle != 0):
                diff_array = Q_interp_p_cycle - avg_initial_partial_Q_for_diff
                capacity_difference_feature = np.mean(np.abs(diff_array))
            else:
                capacity_difference_feature = 0.0

        if FEATURES_TO_INCLUDE['q_profile_stats'] and np.any(Q_interp_p_cycle != 0):
            q_profile_mean = np.mean(Q_interp_p_cycle)
            q_profile_skew = skew(Q_interp_p_cycle) if Q_interp_p_cycle.shape[0] > 2 else 0.0
            q_profile_kurt = kurtosis(Q_interp_p_cycle) if Q_interp_p_cycle.shape[0] > 2 else 0.0
            q_profile_auc = calculate_auc(Q_interp_p_cycle)
        else:
            q_profile_mean, q_profile_skew, q_profile_kurt, q_profile_auc = 0.0, 0.0, 0.0, 0.0

        if FEATURES_TO_INCLUDE['v_profile_stats'] and np.any(V_interp_p_cycle != 0):
            v_profile_mean = np.mean(V_interp_p_cycle)
            v_profile_skew = skew(V_interp_p_cycle) if V_interp_p_cycle.shape[0] > 2 else 0.0
            v_profile_kurt = kurtosis(V_interp_p_cycle) if V_interp_p_cycle.shape[0] > 2 else 0.0
            v_profile_auc = calculate_auc(V_interp_p_cycle)
        else:
            v_profile_mean, v_profile_skew, v_profile_kurt, v_profile_auc = 0.0, 0.0, 0.0, 0.0

        CELL_FOR_Q_PLOT = 9
        CYCLES_FOR_Q_PLOT = [1, 400, 800] 

        if 'q_profiles_for_plotting' not in locals():
            q_profiles_for_plotting = {}

        if cell_num == CELL_FOR_Q_PLOT and i in CYCLES_FOR_Q_PLOT:
            q_profiles_for_plotting[i] = Q_interp_p_cycle

        if (cell_num == CELL_TO_ANALYZE_PLOT_EXAMPLE and i == CYCLE_TO_ANALYZE_PLOT_EXAMPLE):
            plot_example_V_full_raw_interp = V_interp_full_i
            plot_example_Q_partial_features = Q_interp_p_cycle
            plot_example_I_full_raw_interp = I_interp_full_i
            plot_example_t_full_raw_interp = uniform_t_full_interp_i
            plot_example_time_highlight_start_full_range = time_highlight_start_full_range
            plot_example_time_highlight_end_full_range = time_highlight_end_full_range
            plot_example_cell_num = cell_num
            plot_example_cycle_num = i
            plot_example_V_partial_features = V_interp_p_cycle
            plot_example_dQdV_partial_features = DQDV_interp_p_cycle

       
        features_for_this_cycle = []
        # Profile features are NOT added (all set to False in FEATURES_TO_INCLUDE)
        
        # Collect scalar features
        flat_features_for_this_cycle = []
        if FEATURES_TO_INCLUDE['capacity_difference']:
            flat_features_for_this_cycle.append(capacity_difference_feature)
        if FEATURES_TO_INCLUDE['q_profile_stats']:
            flat_features_for_this_cycle.extend([q_profile_mean, q_profile_skew, q_profile_kurt, q_profile_auc])
        if FEATURES_TO_INCLUDE['v_profile_stats']:
            flat_features_for_this_cycle.extend([v_profile_mean, v_profile_skew, v_profile_kurt, v_profile_auc])

        # Since we have no profile features, we create a placeholder
        # This will be replaced later with tiled scalar features
        if not flat_features_for_this_cycle:
            raise ValueError("No scalar features selected. Please enable at least one scalar feature.")
        
        # Create a dummy profile structure (will be replaced with tiled scalars)
        num_scalars = len(flat_features_for_this_cycle)
        dummy_profile = np.zeros((TARGET_LENGTH_PARTIAL_FEATURES, num_scalars))
        current_cell_features.append(dummy_profile)
        

        current_cell_all_features_flattened.append(np.array(flat_features_for_this_cycle))

    if current_cell_features:
        all_cells_features_list.append(np.array(current_cell_features))
    else:
        all_cells_features_list.append(np.array([]))

    if current_cell_all_features_flattened:
        all_cells_all_features_flattened.append(np.array(current_cell_all_features_flattened))
    else:
        all_cells_all_features_flattened.append(np.array([]))


print(f"\n--- Generating Example Plots for Cell {CELL_TO_ANALYZE_PLOT_EXAMPLE}, Cycle {CYCLE_TO_ANALYZE_PLOT_EXAMPLE} ---")
fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE_SUBPLOT[0]*2, FIGSIZE_SUBPLOT[1]), dpi=DPI)
plot_overview_and_sequence(axes[0], axes[1], plot_example_V_full_raw_interp, plot_example_I_full_raw_interp, plot_example_t_full_raw_interp,
                           plot_example_time_highlight_start_full_range, plot_example_time_highlight_end_full_range,
                           plot_example_V_partial_features, plot_example_Q_partial_features,
                           FULL_OPERATING_VOLTAGE_RANGE, FEATURE_EXTRACTION_VOLTAGE_RANGE,
                           plot_example_cycle_num, plot_example_cell_num, CHARGE_OR_DISCHARGE_FOCUS, TARGET_LENGTH_PARTIAL_FEATURES, FEATURES_TO_INCLUDE)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'Cell{plot_example_cell_num}_Cycle{plot_example_cycle_num}_Overview_and_Sequence.png', dpi=DPI)
plt.show()

print("\n--- Generating SOH Degradation and Q Profile Evolution Plots (Q1 style) ---")

fig, (ax_soh, ax_cap) = plt.subplots(
    1, 2,
    figsize=(8.5, 3.2),
    dpi=DPI
)

# ======================================================
# 1) SOH DEGRADATION subplot
# ======================================================
ax_soh.set_facecolor("white")

# Publication-style color palette
cell_colors = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
    "#e6ab02", "#a6761d", "#666666", "#1f78b4", "#b2df8a"
]

min_soh = 100.0
max_cycle = 0

for cell_idx, cell_num in enumerate(CELLS_TO_ANALYZE):
    soh_arr = np.asarray(all_cells_soh_lists_per_cell[cell_idx], dtype=float)
    if soh_arr.size == 0:
        continue

    # Convert to percentage and clip to physical range
    soh_pct = np.clip(soh_arr * 100.0, 0, 100)
    cycles = np.arange(1, soh_pct.size + 1)

    min_soh = min(min_soh, soh_pct.min())
    max_cycle = max(max_cycle, cycles.max())

    color = cell_colors[cell_idx % len(cell_colors)]

    ax_soh.plot(
        cycles,
        soh_pct,
        color=color,
        linewidth=2.0,
        solid_capstyle="round",
        label=f"Cell {cell_num}",
        zorder=2
    )

# 80% SOH threshold
ax_soh.axhline(
    80.0,
    color="0.55",
    linestyle="--",
    linewidth=1.2,
    zorder=1
)

ax_soh.text(
    0.015, 0.055,
    "80% SOH threshold",
    transform=ax_soh.transAxes,
    fontsize=8.8,
    color="0.35"
)

ax_soh.set_xlabel("Cycle Number", fontsize=12)
ax_soh.set_ylabel("SOH (%)", fontsize=12)
ax_soh.set_xlim(0, max_cycle * 1.02)
ax_soh.set_ylim(max(75, min_soh - 2), 100.5)
ax_soh.tick_params(axis="both", direction="in", length=4.5, width=1.0, labelsize=10, pad=4)

leg1 = ax_soh.legend(
    frameon=True,
    facecolor="white",
    edgecolor="0.80",
    framealpha=0.95,
    fontsize=8.5,
    loc="lower right",
    ncol=2,
    borderpad=0.55,
    labelspacing=0.35,
    handlelength=1.4,
    handletextpad=0.45,
    columnspacing=0.9,
    prop={"family": "Times New Roman", "size": 8.5}
)
leg1.get_frame().set_linewidth(0.9)

for spine in ax_soh.spines.values():
    spine.set_linewidth(1.0)

# ======================================================
# 2) CAPACITY PROFILE EVOLUTION subplot
# ======================================================
ax_cap.set_facecolor("white")

cycle_colors = {
    1: "#1f3a93",    # dark blue
    400: "#c44569",  # muted crimson
    800: "#2ca02c",  # green
}

all_q_clean = []

preferred_order = [1, 400, 800]
ordered_cycles = [c for c in preferred_order if c in q_profiles_for_plotting]
ordered_cycles += [c for c in sorted(q_profiles_for_plotting.keys()) if c not in ordered_cycles]

for cycle_num in ordered_cycles:
    q_profile = np.asarray(q_profiles_for_plotting[cycle_num], dtype=float)

    
    q_profile = np.maximum(q_profile, 0.0)

    
    x_idx = np.arange(1, q_profile.size + 1)

    color = cycle_colors.get(cycle_num, "#444444")

    ax_cap.plot(
        x_idx,
        q_profile,
        color=color,
        linewidth=2.6,
        solid_capstyle="round",
        label=f"Cycle {cycle_num}",
        zorder=2
    )

    all_q_clean.append(q_profile)

all_q_vals = np.concatenate(all_q_clean)

ax_cap.set_xlabel("Interpolated Index", fontsize=12)
ax_cap.set_ylabel("Capacity (Ah)", fontsize=12)

# Force full 1–100 display
ax_cap.set_xlim(0, 100)
ax_cap.set_xticks([0, 20, 40, 60, 80, 100])

ax_cap.set_ylim(max(0.0, all_q_vals.min() - 0.02), all_q_vals.max() + 0.015)
ax_cap.tick_params(axis="both", direction="in", length=4.5, width=1.0, labelsize=10, pad=4)

ax_cap.text(
    0.03, 0.94,
    f"Window: {FEATURE_EXTRACTION_VOLTAGE_RANGE[0]:.2f}-{FEATURE_EXTRACTION_VOLTAGE_RANGE[1]:.2f} V",
    transform=ax_cap.transAxes,
    fontsize=8.8,
    color="0.35",
    va="top"
)

leg2 = ax_cap.legend(
    frameon=True,
    facecolor="white",
    edgecolor="0.80",
    framealpha=0.95,
    loc="lower right",
    prop={"family": "Times New Roman", "size": 9}
)
leg2.get_frame().set_linewidth(0.9)

for spine in ax_cap.spines.values():
    spine.set_linewidth(1.0)

# ======================================================
# Panel labels
# ======================================================
fig.text(0.048, 0.02, "(a)", fontsize=12, fontname="Times New Roman", va="top", fontweight="bold")
fig.text(0.525, 0.02, "(b)", fontsize=12, fontname="Times New Roman", va="top", fontweight="bold")

# ======================================================
# Final layout and save
# ======================================================
fig.tight_layout(w_pad=2.0)
fig.subplots_adjust(bottom=0.18)

plt.savefig("SOH_and_Q_Profile_Degradation_Q1.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig("SOH_and_Q_Profile_Degradation_Q1.pdf", bbox_inches="tight", facecolor="white")
plt.savefig("SOH_and_Q_Profile_Degradation_Q1.tif", dpi=600, bbox_inches="tight", facecolor="white")

plt.show()
print("Figure saved.")


# Perform Feature Selection
print("\n--- Starting Feature Selection Procedure ---")
feature_names_list = []
if FEATURES_TO_INCLUDE['capacity_difference']:
    feature_names_list.append('Q_diff_partial_avg')
if FEATURES_TO_INCLUDE['q_profile_stats']:
    feature_names_list.extend(['Q_mean', 'Q_skew', 'Q_kurtosis', 'Q_AUC'])
if FEATURES_TO_INCLUDE['v_profile_stats']:
    feature_names_list.extend(['V_mean', 'V_skew', 'V_kurtosis', 'V_AUC'])

features_to_select_from = np.concatenate([arr for arr in all_cells_all_features_flattened if arr.size > 0], axis=0)
SOH_for_selection = np.concatenate([arr for arr in all_cells_soh_list if arr.size > 0], axis=0)

selected_feature_names, selected_flat_features = perform_feature_selection(
    features=features_to_select_from,
    SOH=SOH_for_selection,
    feature_names=feature_names_list
)
print(f"\nSelected scalar features: {selected_feature_names}")


selected_all_features = []
for cell_idx in range(len(all_cells_all_features_flattened)):
    original_flat_features = all_cells_all_features_flattened[cell_idx]
    if original_flat_features.size > 0:
        # Extract only the selected features
        indices = [feature_names_list.index(name) for name in selected_feature_names]
        selected_flat_for_cell = original_flat_features[:, indices]
        
        # Tile each scalar feature across the time dimension (100 points)
        # This creates pseudo-profiles from scalar features
        num_cycles = selected_flat_for_cell.shape[0]
        num_selected_scalars = selected_flat_for_cell.shape[1]
        
        # Shape: (num_cycles, 100, num_selected_scalars)
        # Each scalar value is repeated 100 times along the time axis
        tiled_features = np.zeros((num_cycles, TARGET_LENGTH_PARTIAL_FEATURES, num_selected_scalars))
        for feat_idx in range(num_selected_scalars):
            for cycle_idx in range(num_cycles):
                tiled_features[cycle_idx, :, feat_idx] = selected_flat_for_cell[cycle_idx, feat_idx]
        
        selected_all_features.append(tiled_features)
        print(f"Cell {cell_idx}: Tiled {num_selected_scalars} scalar features → shape {tiled_features.shape}")
    else:
        selected_all_features.append(np.array([]))


# Re-run scaling with the tiled scalar features
all_features_flat = np.vstack([cell_feats.reshape(-1, cell_feats.shape[-1]) for cell_feats in selected_all_features if cell_feats.size > 0])
scaler_combined_features = MinMaxScaler()
all_features_scaled_flat = scaler_combined_features.fit_transform(all_features_flat)

scaled_cells_features_list = []
current_idx = 0
for cell_feats in selected_all_features:
    if cell_feats.size > 0:
        num_cycles = cell_feats.shape[0]
        num_points = cell_feats.shape[1]
        num_features_actual = cell_feats.shape[2]
        scaled_cell_data = all_features_scaled_flat[current_idx : current_idx + num_cycles * num_points].reshape(num_cycles, num_points, num_features_actual)
        scaled_cells_features_list.append(scaled_cell_data)
        current_idx += num_cycles * num_points
    else:
        scaled_cells_features_list.append(np.array([]))


df_scalar = pd.DataFrame(features_to_select_from, columns=feature_names_list)
df_scalar["SOH"] = SOH_for_selection

# Clean infinities / all-NaN columns (same as before)
df_scalar = df_scalar.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

corr_with_soh = df_scalar.corr()["SOH"].drop("SOH").sort_values(ascending=False)

print("\n--- Generating Scalar Feature–SOH Correlation Heatmap (Q1 style) ---")

corr_df = corr_with_soh.to_frame(name="corr(SOH)")
fig, ax = plt.subplots(figsize=(3.2, 3.6), dpi=DPI)

sns.heatmap(
    corr_df,
    ax=ax,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"shrink": 0.7}
)

#ax.set_title("Correlation of Scalar Features with SOH", fontsize=13, pad=8)
ax.set_xlabel("Correlation", fontsize=11)
#ax.set_ylabel("Scalar Feature", fontsize=11)

# Make y-labels left aligned
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.tight_layout()
plt.savefig("Scalar_Feature_vs_SOH_Correlation_Q1.pdf", dpi=DPI, bbox_inches="tight")
plt.show()

print("\n=== SCALAR-ONLY FEATURE EXTRACTION COMPLETE ===")
print(f"Total scalar features selected: {len(selected_feature_names)}")
print(f"Feature names: {selected_feature_names}")
print(f"Features are tiled {TARGET_LENGTH_PARTIAL_FEATURES} times to create pseudo-profiles for CNN")


