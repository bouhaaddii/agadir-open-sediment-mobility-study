from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "Results/Bathymetry/ATL24_Sentinel2_IBI_SSH_24h/Unique_Photon_Audit"
    / "nearest_scene_unique_photon_oof_predictions.csv.gz"
)
KEYS = ["granule_id", "beam", "source_photon_index"]
RAW = "raw_instantaneous_depth"
ADJUSTED = "IBI_zos_relative_adjustment"
BIN_EDGES = [0, 2, 5, 10, 15, 20, 40]
BIN_LABELS = ["0-2 m", "2-5 m", "5-10 m", "10-15 m", "15-20 m", "20-40 m"]
WINDOWS = ["primary_0_6h", "extended_6_24h", "combined_0_24h"]


def as_source_index(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.all(np.isclose(numeric, np.round(numeric))):
        raise SystemExit("source_photon_index contains missing or non-integer values.")
    return numeric.astype("int64").astype(str)


def effective_group_count(counts: pd.Series) -> float:
    shares = counts.to_numpy(dtype=float) / float(counts.sum())
    return float(1.0 / np.sum(shares**2)) if len(shares) else float("nan")


def skill(model_mse: np.ndarray, baseline_mse: np.ndarray) -> np.ndarray | float:
    model_rmse = np.sqrt(np.mean(model_mse, axis=-1))
    baseline_rmse = np.sqrt(np.mean(baseline_mse, axis=-1))
    return np.divide(
        baseline_rmse - model_rmse,
        baseline_rmse,
        out=np.full_like(np.asarray(model_rmse, dtype=float), np.nan),
        where=baseline_rmse > 0,
    )


def prepare(data: pd.DataFrame) -> pd.DataFrame:
    required = [
        *KEYS, "maximum_grid_distance_km", "window_analysis", "target_scenario", "model",
        "observed_depth_m", "prediction_depth_m", "fold_median_baseline_depth_m",
        "instantaneous_depth_m", "scene_id", "acquisition_date",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise SystemExit(
            "OOF predictions lack stable photon identifiers. Rerun Scripts/18 after updating it; "
            f"missing columns: {missing}"
        )
    data = data.copy()
    data["source_photon_index"] = as_source_index(data["source_photon_index"])
    for column in (
        "maximum_grid_distance_km", "observed_depth_m", "prediction_depth_m",
        "fold_median_baseline_depth_m", "instantaneous_depth_m",
    ):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    valid = np.ones(len(data), dtype=bool)
    for column in (
        "maximum_grid_distance_km", "observed_depth_m", "prediction_depth_m",
        "fold_median_baseline_depth_m", "instantaneous_depth_m",
    ):
        valid &= np.isfinite(data[column].to_numpy(dtype=float))
    valid &= data["instantaneous_depth_m"].between(0, 40, inclusive="left").to_numpy()
    data = data.loc[valid].copy()
    data["raw_depth_bin"] = pd.cut(
        data["instantaneous_depth_m"], bins=BIN_EDGES, labels=BIN_LABELS,
        right=False, include_lowest=True,
    )
    return data.loc[data["raw_depth_bin"].notna()].copy()


def scenario_group_stats(frame: pd.DataFrame) -> pd.DataFrame:
    copy = frame.copy()
    copy["model_error_sq"] = (copy["prediction_depth_m"] - copy["observed_depth_m"]) ** 2
    copy["baseline_error_sq"] = (
        copy["fold_median_baseline_depth_m"] - copy["observed_depth_m"]
    ) ** 2
    return copy.groupby("granule_id", sort=True).agg(
        model_mse=("model_error_sq", "mean"),
        baseline_mse=("baseline_error_sq", "mean"),
        n_photons=("source_photon_index", "size"),
    )


def summarize_pair(
    raw: pd.DataFrame,
    adjusted: pd.DataFrame,
    *, radius: float, window: str, model: str, depth_bin: str,
    replicates: int, rng: np.random.Generator,
) -> dict[str, object]:
    raw_stats = scenario_group_stats(raw)
    adjusted_stats = scenario_group_stats(adjusted)
    common_granules = raw_stats.index.intersection(adjusted_stats.index).sort_values()
    if len(common_granules) != len(raw_stats) or len(common_granules) != len(adjusted_stats):
        raise ValueError("Paired scenario support differs by granule after photon matching.")
    raw_stats = raw_stats.loc[common_granules]
    adjusted_stats = adjusted_stats.loc[common_granules]
    raw_model = raw_stats["model_mse"].to_numpy(dtype=float)
    raw_base = raw_stats["baseline_mse"].to_numpy(dtype=float)
    adj_model = adjusted_stats["model_mse"].to_numpy(dtype=float)
    adj_base = adjusted_stats["baseline_mse"].to_numpy(dtype=float)
    n_groups = len(common_granules)
    if n_groups == 0:
        return {}

    point_raw_skill = float(skill(raw_model, raw_base))
    point_adjusted_skill = float(skill(adj_model, adj_base))
    point_delta_skill = point_adjusted_skill - point_raw_skill
    point_delta_rmse = float(np.sqrt(np.mean(adj_model)) - np.sqrt(np.mean(raw_model)))

    draw = rng.integers(0, n_groups, size=(replicates, n_groups))
    b_raw_skill = skill(raw_model[draw], raw_base[draw])
    b_adj_skill = skill(adj_model[draw], adj_base[draw])
    b_delta_skill = b_adj_skill - b_raw_skill
    b_delta_rmse = np.sqrt(np.mean(adj_model[draw], axis=1)) - np.sqrt(
        np.mean(raw_model[draw], axis=1)
    )
    raw_lo, raw_hi = np.nanquantile(b_raw_skill, [0.025, 0.975])
    adj_lo, adj_hi = np.nanquantile(b_adj_skill, [0.025, 0.975])
    delta_lo, delta_hi = np.nanquantile(b_delta_skill, [0.025, 0.975])
    delta_rmse_lo, delta_rmse_hi = np.nanquantile(b_delta_rmse, [0.025, 0.975])

    counts = raw_stats["n_photons"]
    n_photons = int(counts.sum())
    max_share = float(counts.max() / n_photons)
    eff_groups = effective_group_count(counts)
    balanced = (
        n_photons >= 50 and n_groups >= 5 and eff_groups >= 3 and max_share <= 0.50
    )
    if balanced:
        status = "replicated_and_balanced"
    elif n_groups >= 2:
        status = "sparse_or_unbalanced"
    else:
        status = "single_granule_descriptive_only"

    if delta_lo > 0:
        direction = "higher predictive skill for IBI-adjusted target"
    elif delta_hi < 0:
        direction = "higher predictive skill for raw target"
    else:
        direction = "paired interval includes no difference"
    return {
        "maximum_grid_distance_km": radius,
        "window_analysis": window,
        "model": model,
        "depth_stratum_defined_by": "raw instantaneous ATL24 depth",
        "depth_bin": depth_bin,
        "common_unique_photons": n_photons,
        "common_granules": n_groups,
        "effective_granule_count_by_photon_weight": eff_groups,
        "largest_granule_photon_share": max_share,
        "raw_target_equal_granule_skill": point_raw_skill,
        "raw_target_skill_CI95_low": float(raw_lo),
        "raw_target_skill_CI95_high": float(raw_hi),
        "IBI_adjusted_target_equal_granule_skill": point_adjusted_skill,
        "IBI_adjusted_target_skill_CI95_low": float(adj_lo),
        "IBI_adjusted_target_skill_CI95_high": float(adj_hi),
        "paired_delta_skill_adjusted_minus_raw": point_delta_skill,
        "paired_delta_skill_CI95_low": float(delta_lo),
        "paired_delta_skill_CI95_high": float(delta_hi),
        "paired_delta_model_RMSE_m_adjusted_minus_raw": point_delta_rmse,
        "paired_delta_model_RMSE_CI95_low_m": float(delta_rmse_lo),
        "paired_delta_model_RMSE_CI95_high_m": float(delta_rmse_hi),
        "bootstrap_replicates": replicates,
        "bootstrap_unit": "paired ATL24 granule clusters; identical resamples for both targets",
        "interval_scope": (
            "conditional on common-photon saved OOF predictions; model fitting, datum, IBI, "
            "optical and unsampled-area uncertainty are not propagated"
        ),
        "interpretation": direction,
        "support_status": status,
    }


def make_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    plot = summary.loc[
        summary["maximum_grid_distance_km"].eq(5.0)
        & summary["model"].eq("Multiband_RandomForest")
    ].copy()
    if plot.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), sharey=True, constrained_layout=True)
    colors = {"primary_0_6h": "#19647E", "extended_6_24h": "#E07A32", "combined_0_24h": "#517A61"}
    for ax, window in zip(axes, WINDOWS):
        part = plot.loc[plot["window_analysis"].eq(window)].set_index("depth_bin").reindex(BIN_LABELS)
        supported = part["support_status"].eq("replicated_and_balanced")
        x = part["paired_delta_skill_adjusted_minus_raw"].to_numpy(dtype=float)
        lo = part["paired_delta_skill_CI95_low"].to_numpy(dtype=float)
        hi = part["paired_delta_skill_CI95_high"].to_numpy(dtype=float)
        y = np.arange(len(BIN_LABELS))
        valid = supported.to_numpy() & np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
        ax.errorbar(
            x[valid], y[valid],
            xerr=np.vstack([x[valid] - lo[valid], hi[valid] - x[valid]]),
            fmt="o", color=colors[window], ecolor=colors[window], capsize=3,
            markersize=5, linewidth=1.3,
        )
        if not valid.any():
            ax.text(
                0.5, 0.50, "No depth stratum met\nthe replication criteria",
                ha="center", va="center", transform=ax.transAxes, fontsize=9,
                color="#555555",
            )
        ax.axvline(0, color="#333333", linewidth=0.9, linestyle="--")
        ax.set_title({
            "primary_0_6h": "Near-synchronous (0–6 h)",
            "extended_6_24h": "Time-mismatched (6–24 h)",
            "combined_0_24h": "Combined (0–24 h)",
        }[window], fontsize=10)
        ax.set_xlabel("Paired change in equal-granule skill\n(IBI-adjusted minus raw target)")
        ax.grid(axis="x", alpha=0.2)
        ax.invert_yaxis()
    axes[0].set_yticks(np.arange(len(BIN_LABELS)), BIN_LABELS)
    axes[0].set_ylabel("Depth stratum (defined by raw ATL24 depth)")
    fig.suptitle("Does relative IBI adjustment change out-of-fold SDB skill? (≤5 km)", fontsize=12)
    fig.savefig(output_dir / "Figure_paired_IBI_adjustment_skill_difference.png", dpi=350)
    fig.savefig(output_dir / "Figure_paired_IBI_adjustment_skill_difference.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired granule-bootstrap comparison of raw and relative-IBI SDB targets."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()
    if args.replicates < 200:
        raise SystemExit("Use at least 200 bootstrap replicates.")
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "Paired_Target_Comparison"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if not input_path.is_file():
        raise SystemExit(f"Missing Script 18 OOF predictions: {input_path}")

    data = prepare(pd.read_csv(input_path, low_memory=False))
    data = data.loc[data["target_scenario"].isin([RAW, ADJUSTED])].copy()
    data["scene_id"] = data["scene_id"].astype(str)
    data["acquisition_date"] = data["acquisition_date"].astype(str)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []

    group_cols = ["maximum_grid_distance_km", "window_analysis", "model"]
    for (radius, window, model_name), block in data.groupby(group_cols, sort=True):
        raw = block.loc[block["target_scenario"].eq(RAW)].copy()
        adjusted = block.loc[block["target_scenario"].eq(ADJUSTED)].copy()
        if raw.duplicated(KEYS).any() or adjusted.duplicated(KEYS).any():
            raise SystemExit(
                "Repeated photon keys in OOF target tables; refusing a many-to-many comparison."
            )
        metadata_cols = KEYS + ["scene_id", "acquisition_date", "maximum_grid_distance_km", "window_analysis"]
        raw_meta = raw[metadata_cols].drop_duplicates(KEYS)
        adj_meta = adjusted[metadata_cols].drop_duplicates(KEYS)
        paired_meta = raw_meta.merge(
            adj_meta, on=KEYS, how="inner", suffixes=("_raw", "_adjusted"), validate="one_to_one"
        )
        n_raw, n_adjusted, n_common = len(raw), len(adjusted), len(paired_meta)
        support_rows.append({
            "maximum_grid_distance_km": float(radius),
            "window_analysis": str(window),
            "model": str(model_name),
            "raw_oof_photons": n_raw,
            "IBI_adjusted_oof_photons": n_adjusted,
            "common_unique_photons": n_common,
            "raw_only_photons": n_raw - n_common,
            "IBI_adjusted_only_photons": n_adjusted - n_common,
            "granules_common": int(paired_meta["granule_id"].nunique()),
        })
        if n_common == 0:
            continue
        if not (
            paired_meta["scene_id_raw"].eq(paired_meta["scene_id_adjusted"]).all()
            and paired_meta["acquisition_date_raw"].eq(paired_meta["acquisition_date_adjusted"]).all()
        ):
            raise SystemExit("Raw and adjusted targets do not share the same scene/date for matched photons.")

        raw_cols = KEYS + [
            "observed_depth_m", "prediction_depth_m", "fold_median_baseline_depth_m", "raw_depth_bin"
        ]
        adjusted_cols = KEYS + [
            "observed_depth_m", "prediction_depth_m", "fold_median_baseline_depth_m"
        ]
        paired = raw[raw_cols].merge(
            adjusted[adjusted_cols], on=KEYS, how="inner", suffixes=("_raw", "_adjusted"),
            validate="one_to_one",
        )
        for depth_bin, frame in paired.groupby("raw_depth_bin", observed=True, sort=True):
            raw_frame = pd.DataFrame({
                "granule_id": frame["granule_id"],
                "source_photon_index": frame["source_photon_index"],
                "observed_depth_m": frame["observed_depth_m_raw"],
                "prediction_depth_m": frame["prediction_depth_m_raw"],
                "fold_median_baseline_depth_m": frame["fold_median_baseline_depth_m_raw"],
            })
            adj_frame = pd.DataFrame({
                "granule_id": frame["granule_id"],
                "source_photon_index": frame["source_photon_index"],
                "observed_depth_m": frame["observed_depth_m_adjusted"],
                "prediction_depth_m": frame["prediction_depth_m_adjusted"],
                "fold_median_baseline_depth_m": frame["fold_median_baseline_depth_m_adjusted"],
            })
            record = summarize_pair(
                raw_frame, adj_frame, radius=float(radius), window=str(window),
                model=str(model_name), depth_bin=str(depth_bin), replicates=args.replicates, rng=rng,
            )
            if record:
                rows.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "paired_raw_vs_ibi_depth_skill.csv", index=False)
    pd.DataFrame(support_rows).to_csv(output_dir / "paired_photon_support_audit.csv", index=False)
    if not summary.empty:
        make_figure(summary, output_dir)
    caption = (
        "Figure. Paired change in equal-granule out-of-fold skill for the relative-IBI-adjusted target "
        "minus the raw instantaneous-depth target, using identical ATL24 photons and 2,000 paired "
        "granule-cluster bootstrap replicates. Positive values indicate greater predictive skill "
        "relative to each target's own fold-median baseline; intervals crossing zero indicate that "
        "the paired difference is unresolved. The depth strata are defined from raw ATL24 depth. "
        "Only strata meeting the predeclared replication/balance rule are plotted; sparse strata are "
        "reported in the accompanying table. "
        "The IBI adjustment is a regional-model sensitivity, not an absolute datum correction or "
        "independent bathymetric truth. Results are conditional on saved out-of-fold predictions."
    )
    (output_dir / "figure_caption_EN.txt").write_text(caption + "\n", encoding="utf-8")
    notes = (
        "Comparaison appariée des cibles brutes et corrigées par la variation relative de niveau marin IBI.\n\n"
        "L'analyse utilise les mêmes photons ATL24 pour les deux cibles et rééchantillonne les granules "
        "par paires. Elle mesure si le modèle optique prédit mieux chaque cible par rapport à sa propre "
        "référence médiane de pli. Elle ne démontre pas que la correction IBI améliore la profondeur vraie : "
        "la cible corrigée dépend d'un modèle régional, sans marégraphe local ni raccordement vertical absolu.\n\n"
        "La fenêtre 0–6 h est l'analyse temporelle principale; 6–24 h est une sensibilité avec décalage "
        "temporel. Les intervalles sont conditionnels aux prédictions hors pli enregistrées et ne propagent "
        "pas l'incertitude de réajustement des modèles, de datum, de marée, de couleur de l'eau ni de support spatial.\n"
    )
    (output_dir / "interpretation_notes_FR.txt").write_text(notes, encoding="utf-8")
    print("Paired raw-versus-relative-IBI target sensitivity complete; source data were read only.")
    print(f"Bootstrap replicates: {args.replicates}; seed: {args.seed}")
    print(f"Paired skill table: {output_dir / 'paired_raw_vs_ibi_depth_skill.csv'}")
    print(f"Photon-support audit: {output_dir / 'paired_photon_support_audit.csv'}")
    if not summary.empty:
        print(f"Figures: {output_dir / 'Figure_paired_IBI_adjustment_skill_difference.png'} and PDF")
    print(f"Caption and interpretation: {output_dir / 'figure_caption_EN.txt'}; {output_dir / 'interpretation_notes_FR.txt'}")
    print("This compares predictive agreement with two related labels; it does not validate absolute bathymetry or a datum tie.")


if __name__ == "__main__":
    main()
