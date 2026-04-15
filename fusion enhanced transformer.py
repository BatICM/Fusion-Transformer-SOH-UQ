
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_all_seeds(42)
def sigma_scale_for_target_coverage(
    y, mu, sigma, target_coverage=0.95, z=1.96,
    eps=1e-8, s_min=1e-6, s_max=None
):
    sigma_safe = np.maximum(sigma, eps)
    ratios = np.abs(y - mu) / sigma_safe
    p = np.percentile(ratios, target_coverage * 100.0)
    s = float(p / z)
    if s_max is not None:
        s = np.clip(s, s_min, s_max)
    else:
        s = max(s, s_min)
    return s
def coverage_and_width(y, lower, upper):
    cov = np.mean((y >= lower) & (y <= upper))
    width = np.mean(upper - lower)
    return cov, width
def create_sequences(data_list, target_list, seq_length):
    X_seq, y_seq = [], []
    for cell_data, cell_target in zip(data_list, target_list):
        cell_data_np = cell_data.cpu().numpy() if isinstance(cell_data, torch.Tensor) else cell_data
        cell_target_np = cell_target.cpu().numpy() if isinstance(cell_target, torch.Tensor) else cell_target
        if len(cell_data_np) < seq_length + 1:
            continue
        for i in range(len(cell_data_np) - seq_length):
            X_seq.append(cell_data_np[i:i + seq_length])
            y_seq.append(cell_target_np[i + seq_length])
    if not X_seq:
        return np.array([]), np.array([])
    return np.array(X_seq), np.array(y_seq)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
class ImprovedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.head_weight_proj = nn.Linear(self.d_k, 1)
        self.fusion_projection = nn.Linear(self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("scale", torch.tensor(self.d_k ** 0.5, dtype=torch.float32))
    def forward(self, Q, K, V, mask=None):
        B, S, D = Q.shape
        Q = self.W_q(Q).view(B, S, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, S, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, S, self.nhead, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        head_outputs = torch.matmul(attn, V)
        pooled = head_outputs.mean(dim=2)
        logits = self.head_weight_proj(pooled).squeeze(-1)
        head_weights = torch.softmax(logits, dim=1).view(B, self.nhead, 1, 1)
        fused = (head_outputs * head_weights).sum(dim=1)
        fused = self.fusion_projection(fused)
        out = self.W_o(fused)
        return out
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = ImprovedMultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, mask=src_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(nn.functional.gelu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels, target_length, output_dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channels, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self._feature_size = self._get_conv_output(input_channels, target_length)
        self.fc = nn.Linear(self._feature_size, output_dim)
    def _get_conv_output(self, input_channels, length):
        dummy_input = torch.zeros(1, input_channels, length)
        output = self.conv_block(dummy_input)
        return output.view(output.size(0), -1).size(1)
    def forward(self, x):
        batch_size, seq_len, length, n_features = x.size()
        x = x.view(batch_size * seq_len, length, n_features).permute(0, 2, 1)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(batch_size, seq_len, -1)
class TemporalTransformerWithCNNFusion(nn.Module):
    def __init__(self, input_dim, sequence_length, num_features, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.sequence_length = sequence_length
        self.cnn_feature_extractor = CNNFeatureExtractor(num_features, input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)
        encoder_layers = [CustomTransformerEncoderLayer(d_model, nhead, d_model*4, dropout) for _ in range(num_layers)]
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        self.output_projection = nn.Linear(d_model * sequence_length, 64)
        self.fc = nn.Linear(64, 2)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn_feature_extractor(x)
        x = self.pos_encoder(x)
        for layer in self.transformer_encoder:
            x = layer(x)
        x = x.reshape(batch_size, -1)
        x = self.dropout(torch.relu(self.output_projection(x)))
        output = self.fc(x)
        mu = torch.sigmoid(output[:, 0])
        log_var = output[:, 1]
        return mu, log_var
class BatterySeqDataset(Dataset):
    def __init__(self, X, y, src_ids):
        self.X = X; self.y = y; self.src_ids = src_ids
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.src_ids[idx]
   
def refined_negative_log_likelihood_loss(mu, log_var, target, min_log_var=-5.0, max_log_var=5.0, reg_lambda=0.01):
    log_var = torch.clamp(log_var, min=min_log_var, max=max_log_var)
    loss = 0.5 * torch.mean(torch.exp(-log_var) * (mu - target) ** 2 + log_var)
    reg_loss = reg_lambda * torch.mean(log_var ** 2)
    return loss + reg_loss
def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-4, patience=10):
    criterion = refined_negative_log_likelihood_loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None
    early_stop_counter = 0
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_train = 0.0
        for temporal_data, target in train_loader:
            temporal_data, target = temporal_data.to(device), target.to(device).float()
            optimizer.zero_grad()
            mu, log_var = model(temporal_data)
            loss = criterion(mu, log_var, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_train += loss.item()
        epoch_train_loss = running_train / max(1, len(train_loader))
        train_losses.append(epoch_train_loss)
        # Validation
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for temporal_data, target in val_loader:
                temporal_data, target = temporal_data.to(device), target.to(device).float()
                mu, log_var = model(temporal_data)
                vloss = criterion(mu, log_var, target)
                running_val += vloss.item()
        epoch_val_loss = running_val / max(1, len(val_loader))
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        print(f"Epoch {epoch+1}/{num_epochs} | Train: {epoch_train_loss:.6f} | Val: {epoch_val_loss:.6f}")
    return train_losses, val_losses, best_model_state
def smooth_predictions(predictions, window_size=7):
    if len(predictions) < window_size:
        if len(predictions) >= 3:
            return np.convolve(predictions, np.ones(3)/3, mode='same')
        return predictions
    if window_size % 2 == 0:
        window_size += 1
    polyorder = min(3, window_size - 1)
    return savgol_filter(predictions, window_length=window_size, polyorder=polyorder)

def plot_probabilistic_predictions_grid(
    fold_plot_data,
    dpi=600,
    filename="Probabilistic_SOH_Prediction_AllFolds_Grid_3x2.png"
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Optional smoothing for nicer CI boundaries
    try:
        from scipy.ndimage import gaussian_filter1d
        use_smoothing = True
    except Exception:
        use_smoothing = False

    n_folds = len(fold_plot_data)
    max_folds_to_plot = min(5, n_folds)
    z = 1.96

    # =====================================================
    # Publication-style fold colors
    # =====================================================
    # (a) blue, (b) green, (c) teal, (d) sky blue, (e) olive
    fold_colors = ["#2F5DA8", "#1B9E77", "#17A398", "#4EA5F5", "#9A9D2F"]

    # Better combined-panel color for subplot (f)
    combined_color = "#7A3E9D"   # richer muted purple
    combined_ci_color = "#C7B1D6"

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    # =====================================================
    # Figure / axes
    # =====================================================
    fig, axes = plt.subplots(3, 2, figsize=(8.3, 6.25), dpi=dpi)
    axes = axes.ravel()

    # =====================================================
    # Helper
    # =====================================================
    def _style_axis(ax):
        ax.set_ylim(70, 100)
        ax.tick_params(axis="both", labelsize=9, direction="in", length=3.5, width=1.0)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_linewidth(1.0)

    def _get_split_index(y):
        """
        Detect the big reset/jump location automatically.
        """
        if len(y) < 3:
            return None
        dy = np.diff(y)
        idx = np.argmax(np.abs(dy))
        if np.abs(dy[idx]) > 0.08:  # only draw if jump is clearly significant
            return idx + 1
        return None

    # =====================================================
    # Fold panels: (a)-(e)
    # =====================================================
    for fold_idx in range(max_folds_to_plot):
        fd = fold_plot_data[fold_idx]
        ax = axes[fold_idx]

        y = np.asarray(fd["actuals"], dtype=float)
        mu = np.asarray(fd["means"], dtype=float)
        var = np.asarray(fd["variances"], dtype=float)

        var = np.maximum(var, 1e-10)
        std = np.sqrt(var)
        x = np.arange(len(y))

        color = fold_colors[fold_idx]

        # Convert to %
        y_pct = np.clip(y * 100.0, 0, 100)
        mu_pct = np.clip(mu * 100.0, 0, 100)

        lower = np.clip((mu - z * std) * 100.0, 70, 100)
        upper = np.clip((mu + z * std) * 100.0, 70, 100)

        # Smooth CI edges slightly for cleaner appearance
        if use_smoothing and len(lower) > 10:
            lower = gaussian_filter1d(lower, sigma=1.0)
            upper = gaussian_filter1d(upper, sigma=1.0)

        # CI
        ax.fill_between(
            x, lower, upper,
            color=color,
            alpha=0.11,
            zorder=1
        )

        # Actual and predicted
        ax.plot(
            x, y_pct,
            color="black",
            linewidth=1.7,
            zorder=3
        )
        ax.plot(
            x, mu_pct,
            color=color,
            linestyle="--",
            linewidth=1.5,
            zorder=4
        )

        # Lighter, thinner split line
        split_idx = _get_split_index(y)
        if split_idx is not None:
            ax.axvline(
                split_idx,
                color="0.25",
                linestyle="--",
                linewidth=1.0,
                alpha=0.75,
                zorder=2
            )

        _style_axis(ax)

        # Axis labels only where needed
        if fold_idx in (4, 5):
            ax.set_xlabel("Sample Index", fontsize=10, fontname="Times New Roman")

        if fold_idx % 2 == 0:
            ax.set_ylabel("SOH (%)", fontsize=10, fontname="Times New Roman")

        # Panel labels
        ax.text(
            -0.16, -0.16, panel_labels[fold_idx],
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            fontname="Times New Roman"
        )

    # =====================================================
    # Combined panel: (f)
    # =====================================================
    ax = axes[5]

    all_actuals = np.concatenate([np.asarray(fd["actuals"], dtype=float) for fd in fold_plot_data])
    all_means = np.concatenate([np.asarray(fd["means"], dtype=float) for fd in fold_plot_data])
    all_vars = np.concatenate([np.asarray(fd["variances"], dtype=float) for fd in fold_plot_data])

    all_vars = np.maximum(all_vars, 1e-10)
    all_std = np.sqrt(all_vars)
    x = np.arange(len(all_actuals))

    y_pct = np.clip(all_actuals * 100.0, 0, 100)
    mu_pct = np.clip(all_means * 100.0, 0, 100)

    lower = np.clip((all_means - z * all_std) * 100.0, 70, 100)
    upper = np.clip((all_means + z * all_std) * 100.0, 70, 100)

    if use_smoothing and len(lower) > 10:
        lower = gaussian_filter1d(lower, sigma=1.0)
        upper = gaussian_filter1d(upper, sigma=1.0)

    # Better, richer color for subplot (f)
    ax.fill_between(
        x, lower, upper,
        color=combined_ci_color,
        alpha=0.24,
        zorder=1
    )

    ax.plot(
        x, y_pct,
        color="black",
        linewidth=1.8,
        zorder=3
    )
    ax.plot(
        x, mu_pct,
        color=combined_color,
        linestyle="--",
        linewidth=1.55,
        zorder=4
    )

    # Optional: draw split lines for combined panel too
    # They are lighter so they do not dominate
    jump_idx = np.where(np.abs(np.diff(all_actuals)) > 0.08)[0]
    for ji in jump_idx:
        ax.axvline(
            ji + 1,
            color="0.35",
            linestyle="--",
            linewidth=0.8,
            alpha=0.35,
            zorder=2
        )

    _style_axis(ax)
    ax.set_xlabel("Cycle / sequence index", fontsize=10, fontname="Times New Roman")

    ax.text(
        -0.16, -0.16, panel_labels[5],
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        fontname="Times New Roman"
    )

    # =====================================================
    # Shared legend
    # =====================================================
    legend_handles = [
        Line2D([0], [0], color="black", lw=1.7, label="Actual SOH"),
        Line2D([0], [0], color="0.4", lw=1.5, ls="--", label="Predicted mean"),
        Patch(facecolor="0.8", alpha=0.35, edgecolor="none", label="95% CI"),
    ]

    leg = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=3,
        frameon=True,
        edgecolor="0.85",
        framealpha=0.92,
        fontsize=9,
        prop={"family": "Times New Roman", "size": 9}
    )
    leg.get_frame().set_linewidth(0.9)

    # =====================================================
    # Layout / save
    # =====================================================
    plt.tight_layout(rect=[0, 0.07, 1, 1], w_pad=0.8, h_pad=1.0)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    if filename.lower().endswith(".png"):
        plt.savefig(filename[:-4] + ".pdf", bbox_inches="tight", facecolor="white")
        plt.savefig(filename[:-4] + ".tif", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def plot_probabilistic_scatter_grid(
    fold_plot_data,
    dpi=600,
    filename="Scatter_Probabilistic_Predictions_AllFolds_Grid_3x2.png"
):
    n_folds = len(fold_plot_data)
    max_folds_to_plot = min(5, n_folds)
    z = 1.96

    # Compact layout
    fig, axes = plt.subplots(3, 2, figsize=(7.8, 5.8), dpi=dpi)
    axes = axes.ravel()

    # Light colormap with enough contrast
    cmap_points = plt.cm.YlGnBu

    # --------------------------------------------------
    # Shared error scale in PERCENT
    # --------------------------------------------------
    all_errors_pct = np.concatenate([
        np.abs(np.asarray(fd["means"], dtype=float) - np.asarray(fd["actuals"], dtype=float)) * 100.0
        for fd in fold_plot_data
    ])

    norm = plt.Normalize(vmin=float(all_errors_pct.min()), vmax=float(all_errors_pct.max()))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_points)
    sm.set_array([])

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    def _style_axis(ax):
        ax.set_xlim(75, 100)
        ax.set_ylim(75, 100)
        ax.tick_params(axis="both", labelsize=8.5, direction="in", length=3, width=1.0)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_linewidth(1.0)

    # ==================================================
    # Fold plots: axes[0] to axes[4]
    # ==================================================
    for fold_idx in range(max_folds_to_plot):
        fd = fold_plot_data[fold_idx]
        ax = axes[fold_idx]

        y = np.asarray(fd["actuals"], dtype=float)
        mu = np.asarray(fd["means"], dtype=float)
        var = np.asarray(fd["variances"], dtype=float)

        std = np.sqrt(np.maximum(var, 1e-10))
        errors_pct = np.abs(mu - y) * 100.0

        x_actual = np.clip(y * 100.0, 75, 100)
        y_pred = np.clip(mu * 100.0, 75, 100)

        lower = np.clip((mu - z * std) * 100.0, 75, 100)
        upper = np.clip((mu + z * std) * 100.0, 75, 100)

        yerr_lower = y_pred - lower
        yerr_upper = upper - y_pred

        # Light CI bars
        ax.errorbar(
            x_actual,
            y_pred,
            yerr=[yerr_lower, yerr_upper],
            fmt="none",
            ecolor="0.90",
            alpha=0.25,
            linewidth=0.6,
            zorder=1,
            capsize=0
        )

        # Scatter points colored by absolute error (%)
        ax.scatter(
            x_actual,
            y_pred,
            c=errors_pct,
            cmap=cmap_points,
            norm=norm,
            s=8,
            alpha=0.70,
            edgecolors="none",
            zorder=2,
        )

        # 1:1 line
        ax.plot(
            [75, 100], [75, 100],
            color="black",
            linestyle="--",
            lw=1.5,
            alpha=0.9,
            zorder=3
        )

        _style_axis(ax)

        if fold_idx in (4, 5):
            ax.set_xlabel("Actual SOH (%)", fontsize=9, fontname="Times New Roman")

        if fold_idx % 2 == 0:
            ax.set_ylabel("Predicted SOH (%)", fontsize=9, fontname="Times New Roman")

        ax.text(
            -0.15, -0.15, panel_labels[fold_idx],
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            fontname="Times New Roman"
        )

    # ==================================================
    # Combined panel: axes[5]
    # ==================================================
    ax = axes[5]

    all_y = np.concatenate([np.asarray(fd["actuals"], dtype=float) for fd in fold_plot_data])
    all_mu = np.concatenate([np.asarray(fd["means"], dtype=float) for fd in fold_plot_data])
    all_var = np.concatenate([np.asarray(fd["variances"], dtype=float) for fd in fold_plot_data])

    std = np.sqrt(np.maximum(all_var, 1e-10))
    all_err_pct = np.abs(all_mu - all_y) * 100.0   # <-- fixed

    x_actual = np.clip(all_y * 100.0, 75, 100)
    y_pred = np.clip(all_mu * 100.0, 75, 100)

    lower = np.clip((all_mu - z * std) * 100.0, 75, 100)
    upper = np.clip((all_mu + z * std) * 100.0, 75, 100)

    yerr_lower = y_pred - lower
    yerr_upper = upper - y_pred

    ax.errorbar(
        x_actual,
        y_pred,
        yerr=[yerr_lower, yerr_upper],
        fmt="none",
        ecolor="0.88",
        alpha=0.25,
        linewidth=0.6,
        zorder=1,
        capsize=0
    )

    ax.scatter(
        x_actual,
        y_pred,
        c=all_err_pct,
        cmap=cmap_points,
        norm=norm,
        s=8,
        alpha=0.70,
        edgecolors="none",
        zorder=2,
    )

    ax.plot(
        [75, 100], [75, 100],
        color="black",
        linestyle="--",
        lw=1.5,
        alpha=0.9,
        zorder=3
    )

    _style_axis(ax)
    ax.set_xlabel("Actual SOH (%)", fontsize=9, fontname="Times New Roman")

    ax.text(
        -0.15, -0.15, panel_labels[5],
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        fontname="Times New Roman"
    )

    # ==================================================
    # Shared legend
    # ==================================================
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="black", markersize=4, label="Samples"),
        Line2D([0], [0], color="black", linestyle="--", lw=1.5, label="1:1 line"),
        Patch(facecolor="0.90", edgecolor="none", label="95% CI")
    ]

    leg = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.42, 0.01),
        ncol=3,
        frameon=True,
        edgecolor="0.85",
        framealpha=0.92,
        fontsize=8.5,
        prop={"family": "Times New Roman", "size": 8.5}
    )
    leg.get_frame().set_linewidth(0.9)

    # ==================================================
    # Colorbar outside, unit in %
    # ==================================================
    cbar_ax = fig.add_axes([0.90, 0.22, 0.025, 0.56])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Absolute error (%)", fontsize=9, fontname="Times New Roman")
    cbar.ax.tick_params(labelsize=8)

    # ==================================================
    # Layout / save
    # ==================================================
    plt.tight_layout(rect=[0, 0.05, 0.88, 1.0])

    plt.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    if filename.lower().endswith(".png"):
        plt.savefig(filename[:-4] + ".pdf", bbox_inches="tight", facecolor="white")
        plt.savefig(filename[:-4] + ".tif", dpi=dpi, bbox_inches="tight", facecolor="white")

    plt.show()


def kfold_dual_boxplots_paper(
    fold_results,
    coverage_rates,
    mean_interval_widths,
    filename="KFold_Dual_Boxplots_Paper.png",
    dpi=600,
    figsize=(11, 4.5),
    percentify_mae_rmse=True,
    show_fliers=True, # unused, kept for compatibility
    whis=1.5 # unused, kept for compatibility
):
    mae = np.array([fr['mae'] for fr in fold_results], dtype=float)
    rmse = np.array([fr['rmse'] for fr in fold_results], dtype=float)
    r2 = np.array([fr['r2'] for fr in fold_results], dtype=float)
    cov = np.array(coverage_rates, dtype=float) * 100.0
    miw = np.array(mean_interval_widths, dtype=float)
    if percentify_mae_rmse:
        mae *= 100.0
        rmse *= 100.0
    mae_m, mae_s = np.nanmean(mae), np.nanstd(mae)
    rmse_m, rmse_s = np.nanmean(rmse), np.nanstd(rmse)
    r2_m, r2_s = np.nanmean(r2), np.nanstd(r2)
    cov_m, cov_s = np.nanmean(cov), np.nanstd(cov)
    miw_m, miw_s = np.nanmean(miw), np.nanstd(miw)
    colors_left = plt.cm.Blues(np.linspace(0.6, 0.9, 2))
    color_r2 = plt.cm.Greens(0.7)
    colors_right = plt.cm.Purples(np.linspace(0.6, 0.9, 2))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    # ---------- LEFT: deterministic ----------
    axL2 = axL.twinx()
    x_mae_rmse = np.array([0, 1])
    x_r2 = np.array([2])
    width = 0.55
    axL.bar(
        x_mae_rmse[0], mae_m, width,
        yerr=mae_s,
        color=colors_left[0], edgecolor="black", linewidth=1.0,
        capsize=4, label="MAE"
    )
    axL.bar(
        x_mae_rmse[1], rmse_m, width,
        yerr=rmse_s,
        color=colors_left[1], edgecolor="black", linewidth=1.0,
        capsize=4, label="RMSE"
    )
    axL2.bar(
        x_r2[0], r2_m, width,
        yerr=r2_s,
        color=color_r2, edgecolor="black", linewidth=1.0,
        capsize=4, label="R²"
    )
    axL.set_xticks([0, 1, 2])
    axL.set_xticklabels(["MAE", "RMSE", "R²"], fontsize=12, fontname='Times New Roman')
    axL.set_ylabel("MAE / RMSE (%)", fontsize=12, fontname='Times New Roman')
    axL2.set_ylabel("R²", fontsize=12, fontname='Times New Roman')
    axL.tick_params(axis='y', labelsize=12)
    axL2.tick_params(axis='y', labelsize=12)
    axL.grid(axis='y', linestyle=':', alpha=0.35)
    left_all = np.concatenate([mae, rmse])
    if np.isfinite(left_all).any():
        low = max(0, np.nanmin(left_all)*0.9)
        high = np.nanmax(left_all)*1.15
        axL.set_ylim(low, high)
    r2_lo, r2_hi = np.nanmin(r2), np.nanmax(r2)
    pad = (r2_hi - r2_lo)*0.05 if r2_hi > r2_lo else 0.002
    axL2.set_ylim(r2_lo - pad, r2_hi + pad)
    handles = [
        Patch(facecolor=colors_left[0], edgecolor="black", label="MAE"),
        Patch(facecolor=colors_left[1], edgecolor="black", label="RMSE"),
        Patch(facecolor=color_r2, edgecolor="black", label="R²")
    ]
    axL.legend(handles=handles, loc="upper left", fontsize=12,
               frameon=True, framealpha=0.95, prop={'family': 'Times New Roman'})
    # ---------- RIGHT: uncertainty ----------
    axR2 = axR.twinx()
    x_cov_miw = np.array([0, 1])
    axR.bar(
        x_cov_miw[0], cov_m, width,
        yerr=cov_s,
        color=colors_right[0], edgecolor="black", linewidth=1.0,
        capsize=4, label="Coverage rate"
    )
    axR2.bar(
        x_cov_miw[1], miw_m, width,
        yerr=miw_s,
        color=colors_right[1], edgecolor="black", linewidth=1.0,
        capsize=4, label="Interval width"
    )
    axR.set_xticks([0, 1])
    axR.set_xticklabels(["Coverage rate", "Interval width"], fontsize=12, fontname='Times New Roman')
    axR.set_ylabel("Coverage rate (%)", fontsize=12, fontname='Times New Roman')
    axR2.set_ylabel("Average interval width", fontsize=12, fontname='Times New Roman')
    axR.tick_params(axis='y', labelsize=12)
    axR2.tick_params(axis='y', labelsize=12)
    axR.grid(axis='y', linestyle=':', alpha=0.35)
    cov_lo, cov_hi = np.nanmin(cov), np.nanmax(cov)
    cov_pad = (cov_hi - cov_lo)*0.05 if cov_hi > cov_lo else 0.5
    axR.set_ylim(cov_lo - cov_pad, cov_hi + cov_pad)
    miw_lo, miw_hi = np.nanmin(miw), np.nanmax(miw)
    miw_pad = (miw_hi - miw_lo)*0.08 if miw_hi > miw_lo else 0.002
    axR2.set_ylim(miw_lo - miw_pad, miw_hi + miw_pad)
    handles_unc = [
        Patch(facecolor=colors_right[0], edgecolor="black",
              label="Coverage rate"),
        Patch(facecolor=colors_right[1], edgecolor="black",
              label="Interval width")
    ]
    axR.legend(handles=handles_unc, loc="upper left", fontsize=12,
               frameon=True, framealpha=0.95, prop={'family': 'Times New Roman'})
    # Add labels (a) and (b) outside bottom-left
    fig.text(0.050, 0.02, '(a)', fontsize=12, fontname='Times New Roman', va='top', fontweight="bold")
    fig.text(0.525, 0.02, '(b)', fontsize=12, fontname='Times New Roman', va='top', fontweight="bold")
    for ax in (axL, axL2, axR, axR2):
        for s in ax.spines.values():
            s.set_linewidth(1.0)
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()
def run_kfold_training_and_evaluation(
    X_data_tensor_list,
    y_data_tensor_list,
    model_class,
    sequence_length,
    num_features_per_point,
    d_model,
    nhead,
    num_layers,
    dropout,
    num_epochs=50,
    lr=0.0001,
    patience=10,
    n_splits=5,
    batch_size=32,
    model_name="Model"
):
    print(f"\n--- Starting {n_splits}-Fold Cross-Validation for {model_name} ---")
    num_cells = len(X_data_tensor_list)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    all_fold_val_predictions = []
    all_fold_val_actuals = []
    all_fold_val_variances = []
    coverage_rates = []
    mean_interval_widths = []
    # NEW: store raw arrays per fold for the 6-subplot figures
    fold_plot_data = []
    for fold, (train_cell_indices, val_cell_indices) in enumerate(kf.split(range(num_cells))):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        X_train_raw = [X_data_tensor_list[i] for i in train_cell_indices]
        y_train_raw = [y_data_tensor_list[i] for i in train_cell_indices]
        X_val_raw = [X_data_tensor_list[i] for i in val_cell_indices]
        y_val_raw = [y_data_tensor_list[i] for i in val_cell_indices]
        X_train_seq_np, y_train_seq_np = create_sequences(X_train_raw, y_train_raw, sequence_length)
        X_val_seq_np, y_val_seq_np = create_sequences(X_val_raw, y_val_raw, sequence_length)
        if X_train_seq_np.size == 0 or y_train_seq_np.size == 0:
            print(f"Warning: Fold {fold+1} has no valid training sequences. Skipping fold.")
            continue
        if X_val_seq_np.size == 0 or y_val_seq_np.size == 0:
            print(f"Warning: Fold {fold+1} has no valid validation sequences. Skipping fold.")
            continue
        X_train_seq_torch = torch.tensor(X_train_seq_np, dtype=torch.float32).to(device)
        y_train_seq_torch = torch.tensor(y_train_seq_np, dtype=torch.float32).to(device)
        X_val_seq_torch = torch.tensor(X_val_seq_np, dtype=torch.float32).to(device)
        y_val_seq_torch = torch.tensor(y_val_seq_np, dtype=torch.float32).to(device)
        train_dataset = TensorDataset(X_train_seq_torch, y_train_seq_torch)
        val_dataset = TensorDataset(X_val_seq_torch, y_val_seq_torch)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        input_dim = TARGET_LENGTH_PARTIAL_FEATURES
        model = model_class(
            input_dim=input_dim,
            sequence_length=sequence_length,
            num_features=num_features_per_point,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
        train_losses, val_losses, best_model_state = train_model(
            model, train_loader, val_loader,
            num_epochs=num_epochs, lr=lr, patience=patience
        )
        model.load_state_dict(best_model_state)
        model.eval()
        fold_val_predictions = []
        fold_val_actuals = []
        fold_val_variances = []
        with torch.no_grad():
            for temporal_data, target in val_loader:
                temporal_data, target = temporal_data.to(device), target.to(device)
                mu, log_var = model(temporal_data)
                fold_val_predictions.extend(mu.cpu().numpy())
                fold_val_actuals.extend(target.cpu().numpy())
                fold_val_variances.extend(torch.exp(log_var).cpu().numpy())
        fold_val_predictions = np.clip(np.array(fold_val_predictions), 0.0, 1.0)
        fold_val_actuals = np.array(fold_val_actuals)
        fold_val_variances = np.array(fold_val_variances)
        # --- Uncertainty calibration for this fold ---
        sigma_raw = np.sqrt(np.maximum(fold_val_variances, 1e-12))
        s = sigma_scale_for_target_coverage(
            y=fold_val_actuals,
            mu=fold_val_predictions,
            sigma=sigma_raw,
            target_coverage=0.95,
            z=1.96,
            s_max=3.0
        )
        sigma_cal = sigma_raw * s
        fold_val_variances_cal = sigma_cal ** 2
        # coverage + interval width (for your metrics)
        z = 1.96
        lower = fold_val_predictions - z * sigma_cal
        upper = fold_val_predictions + z * sigma_cal
        coverage_rate, mean_interval_width = coverage_and_width(
            fold_val_actuals, lower, upper
        )
        coverage_rates.append(coverage_rate)
        mean_interval_widths.append(mean_interval_width)
        # deterministic metrics
        fold_mae = mean_absolute_error(fold_val_actuals, fold_val_predictions)
        fold_mse = mean_squared_error(fold_val_actuals, fold_val_predictions)
        fold_rmse = np.sqrt(fold_mse)
        fold_r2 = r2_score(fold_val_actuals, fold_val_predictions)
        fold_results.append({
            "mae": fold_mae,
            "mse": fold_mse,
            "rmse": fold_rmse,
            "r2": fold_r2
        })
        all_fold_val_predictions.extend(fold_val_predictions)
        all_fold_val_actuals.extend(fold_val_actuals)
        all_fold_val_variances.extend(fold_val_variances_cal)
        # store raw values for the grid plots (we'll merge/sort inside plotting)
        fold_plot_data.append({
            "actuals": fold_val_actuals,
            "means": fold_val_predictions,
            "variances": fold_val_variances_cal
        })
        print(f"Fold {fold+1} Metrics:")
        print(f" MAE: {fold_mae:.6f} | MSE: {fold_mse:.6f} | RMSE: {fold_rmse:.6f} | R²: {fold_r2:.6f}")
        print(f" Coverage: {coverage_rate:.6f} | Mean Interval Width: {mean_interval_width:.6f}")
    # convert accumulators to arrays
    all_fold_val_predictions = np.array(all_fold_val_predictions)
    all_fold_val_actuals = np.array(all_fold_val_actuals)
    all_fold_val_variances = np.array(all_fold_val_variances)
    # average metrics across folds
    avg_mae = np.mean([res["mae"] for res in fold_results])
    avg_mse = np.mean([res["mse"] for res in fold_results])
    avg_rmse = np.mean([res["rmse"] for res in fold_results])
    avg_r2 = np.mean([res["r2"] for res in fold_results])
    avg_cov = np.mean(coverage_rates)
    avg_miw = np.mean(mean_interval_widths)
    print(f"\n--- Average {model_name} Metrics across {len(fold_results)} Folds ---")
    print(f" Average MAE: {avg_mae:.6f}")
    print(f" Average MSE: {avg_mse:.6f}")
    print(f" Average RMSE: {avg_rmse:.6f}")
    print(f" Average R²: {avg_r2:.6f}")
    print(f" Average Coverage: {avg_cov:.6f}")
    print(f" Average MIW: {avg_miw:.6f}")
    # === NEW: 2 FIGURES WITH 6 SUBPLOTS EACH ===
    plot_probabilistic_predictions_grid(
        fold_plot_data,
        dpi=DPI,
        filename=f"{model_name.replace(' ', '_')}_Probabilistic_SOH_Prediction_Grid.png"
    )
    plot_probabilistic_scatter_grid(
        fold_plot_data,
        dpi=DPI,
        filename=f"{model_name.replace(' ', '_')}_Scatter_Probabilistic_Predictions_Grid.png"
    )
    # your existing dual boxplot summary stays the same
    kfold_dual_boxplots_paper(
        fold_results=fold_results,
        coverage_rates=coverage_rates,
        mean_interval_widths=mean_interval_widths,
        filename=f"{model_name.replace(' ', '_')}_KFold_Dual_Boxplots_Paper.png"
    )
    return {
        "mae": avg_mae,
        "mse": avg_mse,
        "rmse": avg_rmse,
        "r2": avg_r2,
        "coverage_rate": avg_cov,
        "mean_interval_width": avg_miw
    }, all_fold_val_predictions, all_fold_val_actuals
# ===============================================================
# Main Execution
# ===============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    X_data_tensor_list = [
        torch.tensor(cell_feats, dtype=torch.float32)
        for cell_feats in scaled_cells_features_list
        if cell_feats.size > 0
    ]
    y_data_tensor_list = [
        torch.tensor(cell_soh, dtype=torch.float32)
        for cell_soh in all_cells_soh_list
        if cell_soh.size > 0
    ]
    TARGET_LENGTH_PARTIAL_FEATURES = X_data_tensor_list[0].shape[1]
    print(f"\n=== FINAL DATA SUMMARY ===")
    print(f"Number of cells with data: {len(X_data_tensor_list)}")
    if len(X_data_tensor_list) > 0:
        print(f"Example X_data_tensor_list[0] shape: {X_data_tensor_list[0].shape}")
        print(f"Example y_data_tensor_list[0] shape: {y_data_tensor_list[0].shape}")
        print(f"Number of scalar features (tiled): {X_data_tensor_list[0].shape[-1]}")
        print(f"Tiled length per feature: {X_data_tensor_list[0].shape[1]}")
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
    best_fusion_params = FUSION_TRANSFORMER_PARAMS.copy()
    num_features_per_point_final = scaled_cells_features_list[0].shape[-1] if scaled_cells_features_list else 1
    print(f"\n--- Best Hyperparameters Used for CNN-Transformer with SCALAR FEATURES ONLY ---")
    print(f"Hyperparameters: {best_fusion_params}")
    print(f"Number of scalar features (tiled as pseudo-profiles): {num_features_per_point_final}")
    fusion_metrics_kfold, fusion_predictions_kfold, fusion_actuals_kfold = run_kfold_training_and_evaluation(
        X_data_tensor_list,
        y_data_tensor_list,
        model_class=TemporalTransformerWithCNNFusion,
        num_features_per_point=num_features_per_point_final,
        model_name="CNN-Transformer with Scalar Features Only",
        **best_fusion_params
    )
    print("\n=== FINAL MODEL PERFORMANCE (Scalar Features Only) ===")
    print("CNN-Transformer Hybrid (Scalar Features Only):")
    print(f" Average MAE: {fusion_metrics_kfold['mae']:.6f}")
    print(f" Average RMSE: {fusion_metrics_kfold['rmse']:.6f}")
    print(f" Average R² Score: {fusion_metrics_kfold['r2']:.6f}")
    print(f" Average Coverage Rate: {fusion_metrics_kfold['coverage_rate']:.6f}")
    print(f" Average Mean Interval Width: {fusion_metrics_kfold['mean_interval_width']:.6f}")
    print("\n=== SUCCESS: Model trained using ONLY scalar features from partial voltage range (3.45-3.55V) ===")
