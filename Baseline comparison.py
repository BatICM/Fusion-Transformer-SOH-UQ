# =========================
#  Baselines comparrisons
# =========================
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # <-- add this

def _flatten_sequences(X_seq, reduce='mean'):
    """
    X_seq: (N, seq_len, length, n_feat) when using your tiled scalars.
    Turn it into (N, seq_len*n_feat) without the fake 100-length axis.
    """
    if X_seq.ndim == 4:
        if reduce == 'mean':      # average over the 100 tiled steps
            Xr = X_seq.mean(axis=2)         # (N, seq_len, n_feat)
        elif reduce == 'last':    # or just take the last step
            Xr = X_seq[:, :, -1, :]         # (N, seq_len, n_feat)
        else:                     # fallback to full flatten
            Xr = X_seq.reshape(X_seq.shape[0], -1)
            return Xr
        return Xr.reshape(Xr.shape[0], -1)   # (N, seq_len*n_feat)
    else:
        # already flat (e.g., true scalar sequences)
        return X_seq.reshape(X_seq.shape[0], -1)


def _evaluate_interval_metrics(y_true, y_pred, lower, upper):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
    cov  = np.mean((y_true >= lower) & (y_true <= upper))
    miw  = np.mean(upper - lower)
    return mae, rmse, r2, cov, miw

def _split_conformal_intervals(model, X_train, y_train, X_cal, y_cal, X_test, alpha=0.05):
    """
    Split conformal for symmetric intervals around point predictions.
    1) fit on X_train
    2) get residuals on X_cal -> q = quantile(1 - alpha) of |residual|
    3) intervals on X_test: [mu - q, mu + q]
    """
    model.fit(X_train, y_train)
    mu_cal = model.predict(X_cal)
    cal_resid = np.abs(y_cal - mu_cal)
    q = np.quantile(cal_resid, 1 - alpha)
    mu_test = model.predict(X_test)
    lower = mu_test - q
    upper = mu_test + q
    return mu_test, lower, upper, q

def run_kfold_light_baselines(
    X_data_tensor_list,
    y_data_tensor_list,
    sequence_length=5,
    n_splits=5,
    random_state=42,
    alpha=0.05,      
    cal_size=0.2     
):
    print(f"\n=== Lightweight Baseline Benchmark: {n_splits}-Fold (cell-wise) ===")

    model_builders = {
        "Quantile GB (Conformal)": lambda: GradientBoostingRegressor(
            loss="quantile", alpha=0.50, random_state=42
        ),  
        "Lasso (Conformal)": lambda: make_pipeline(
    StandardScaler(with_mean=False),  # sparse-friendly
    Lasso(alpha=1e-3, random_state=42, max_iter=2000)
),
"kNN-15 dist (Conformal)": lambda: make_pipeline(
    StandardScaler(),
    KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1)
),

        "Decision Tree (Conformal)": lambda: DecisionTreeRegressor(
            random_state=42, max_depth=12, min_samples_leaf=5
        ),
        "Random Forest x100 (Conformal)": lambda: RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }

    results = {name: {"fold_metrics": [], "all_pred": [], "all_true": [], "all_width": []}
               for name in model_builders}

    # K-fold across cells 
    num_cells = len(X_data_tensor_list)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_cells, val_cells) in enumerate(kf.split(range(num_cells)), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")

        X_train_raw = [X_data_tensor_list[i] for i in train_cells]
        y_train_raw = [y_data_tensor_list[i] for i in train_cells]
        X_val_raw   = [X_data_tensor_list[i] for i in val_cells]
        y_val_raw   = [y_data_tensor_list[i] for i in val_cells]

        Xtr, ytr = create_sequences(X_train_raw, y_train_raw, sequence_length)
        Xva, yva = create_sequences(X_val_raw,   y_val_raw,   sequence_length)
        if Xtr.size == 0 or Xva.size == 0:
            print("  (No valid sequences in this fold; skipping.)")
            continue

        Xtr_f = _flatten_sequences(Xtr)
        Xva_f = _flatten_sequences(Xva)

        # split-conformal split on the training sequences (no leakage)
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            Xtr_f, ytr, test_size=cal_size, random_state=random_state, shuffle=True
        )

        # run each baseline
        for name, builder in model_builders.items():
            model = builder()
            mu, lo, hi, q = _split_conformal_intervals(model, X_fit, y_fit, X_cal, y_cal, Xva_f, alpha=alpha)
            mae, rmse, r2, cov, miw = _evaluate_interval_metrics(yva, mu, lo, hi)
            results[name]["fold_metrics"].append((mae, rmse, r2, cov, miw))
            results[name]["all_pred"].extend(mu)
            results[name]["all_true"].extend(yva)
            results[name]["all_width"].extend(hi - lo)
            print(f"  {name:27s} | MAE {mae:.5f} | RMSE {rmse:.5f} | R² {r2:.5f} | Cov {cov:.4f} | MIW {miw:.5f} (q={q:.5f})")

    # summarize
    print("\n=== Lightweight Baseline Averages over folds ===")
    summaries = {}
    for name, bag in results.items():
        if not bag["fold_metrics"]:
            continue
        arr = np.array(bag["fold_metrics"])  # (folds, 5)
        avg_mae, avg_rmse, avg_r2, avg_cov, avg_miw = arr.mean(axis=0)
        summaries[name] = dict(
            mae=avg_mae, rmse=avg_rmse, r2=avg_r2,
            coverage_rate=avg_cov, mean_interval_width=avg_miw
        )
        print(f"\n{name}:")
        print(f" Average MAE: {avg_mae:.6f}")
        print(f" Average RMSE: {avg_rmse:.6f}")
        print(f" Average R²: {avg_r2:.6f}")
        print(f" Average Coverage: {avg_cov:.6f}")
        print(f" Average MIW: {avg_miw:.6f}")
    return summaries

if __name__ == "__main__":
    light_baseline_summaries = run_kfold_light_baselines(
        X_data_tensor_list=X_data_tensor_list,
        y_data_tensor_list=y_data_tensor_list,
        sequence_length=FUSION_TRANSFORMER_PARAMS['sequence_length'],
        n_splits=5,
        random_state=42,
        alpha=0.05,     
        cal_size=0.2     
    )

    print("\n=== Your Model (reported) ===")
    print(" Average MAE: 0.003673")
    print(" Average MSE: 0.000025")
    print(" Average RMSE: 0.005008")
    print(" Average R²: 0.990871")
    print(" Average Coverage: 0.935571")
    print(" Average MIW: 0.032512")

    print("\n=== Lightweight Baseline Summaries ===")
    for k, v in light_baseline_summaries.items():
        print(f"{k}: {v}")

