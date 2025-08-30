from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_poisson_deviance

PROC = Path("data/processed/french.parquet")
PREDS = Path("data/interim/freq_preds.parquet")
OUTD = Path("data/interim/freq_analysis")
OUTD.mkdir(parents=True, exist_ok=True)

MODEL_RATE_COLS = {
    "poisson": "rate_poisson",
    "negbin":  "rate_negbin",
    "gbm":     "rate_gbm",
}
CAT_COLS = ["vehicle_brand", "fuel", "area", "region"]
NUM_COLS = ["driver_age", "vehicle_age", "vehicle_power", "bonus_malus", "density"]

# ---------- helpers ----------
def _ensure_cols(df: pd.DataFrame):
    need = {"policy_id","exposure","claim_count","yhat_poisson","yhat_negbin","yhat_gbm","split"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"Brakuje kolumn w freq_preds: {miss}")

def _add_rates(df: pd.DataFrame) -> pd.DataFrame:
    expo = np.clip(df["exposure"].to_numpy(float), 1e-9, None)
    df["rate_obs"]     = df["claim_count"].astype(float) / expo
    df["rate_poisson"] = np.clip(df["yhat_poisson"].astype(float) / expo, 1e-12, None)
    df["rate_negbin"]  = np.clip(df["yhat_negbin"].astype(float)  / expo, 1e-12, None)
    df["rate_gbm"]     = np.clip(df["yhat_gbm"].astype(float)     / expo, 1e-12, None)
    return df

def poisson_dev(y, mu):
    mu = np.clip(np.asarray(mu, float), 1e-12, None)
    y = np.asarray(y, float)
    return mean_poisson_deviance(y, mu)

def lorenz_points(y, w, score, ascending=True):
    df = pd.DataFrame({"y": y, "w": w, "score": score}).copy()
    df = df.sort_values("score", ascending=ascending, kind="mergesort")
    wsum = df["w"].sum()
    ysum = df["y"].sum()
    cx = (df["w"].cumsum() / wsum).to_numpy()
    cy = (df["y"].cumsum() / max(ysum, 1e-12)).to_numpy()
    cx = np.concatenate([[0.0], cx])
    cy = np.concatenate([[0.0], cy])
    return cx, cy

def gini_from_lorenz(cx, cy):
    area = np.trapz(cy, cx)
    return float(1.0 - 2.0 * area)

def gains_table(df: pd.DataFrame, rate_col: str, bins=(0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0)):
    d = df.sort_values(rate_col, ascending=False)
    d["cum_expo"] = d["exposure"].cumsum()
    d["cum_clms"] = d["claim_count"].cumsum()
    tot_e = d["exposure"].sum()
    tot_c = d["claim_count"].sum()
    rows = []
    for p in bins:
        cut_e = tot_e * p
        sub = d.loc[d["cum_expo"] <= cut_e]
        expo_share = sub["exposure"].sum() / tot_e if tot_e > 0 else 0.0
        claim_share = sub["claim_count"].sum() / max(tot_c, 1e-12)
        rows.append((p, expo_share, claim_share))
    return pd.DataFrame(rows, columns=["threshold_expo_share","expo_share","claim_share"])

def decile_calibration(df: pd.DataFrame, rate_col: str, n=10):
    # decyle po ekspozycji (równe wiadra ekspozycyjne)
    d = df[["exposure","claim_count",rate_col]].copy().sort_values(rate_col, ascending=True)
    cum_e = d["exposure"].cumsum()
    tot_e = d["exposure"].sum()
    edges = [tot_e * i / n for i in range(1, n)]
    d["bin"] = np.searchsorted(edges, cum_e, side="right")
    grp = d.groupby("bin").apply(
        lambda g: pd.Series({
            "exposure": g["exposure"].sum(),
            "observed_rate": g["claim_count"].sum()/max(g["exposure"].sum(),1e-12),
            "pred_rate": np.average(g[rate_col], weights=g["exposure"]),
            "n": len(g)
        })
    ).reset_index().sort_values("bin")
    grp["abs_err"] = (grp["observed_rate"] - grp["pred_rate"]).abs()
    return grp

def topk_table(df: pd.DataFrame, rate_col: str, k_list=(0.05,0.1,0.2,0.3,0.5)):
    d = df.sort_values(rate_col, ascending=False)
    d["cum_expo_share"] = d["exposure"].cumsum() / d["exposure"].sum()
    d["cum_claim_share"] = d["claim_count"].cumsum() / max(d["claim_count"].sum(), 1e-12)
    rows = []
    for k in k_list:
        sub = d[d["cum_expo_share"] <= k]
        rows.append({
            "k_expo": k,
            "expo": sub["exposure"].sum(),
            "claims": sub["claim_count"].sum(),
            "claim_share": sub["claim_count"].sum()/max(d["claim_count"].sum(),1e-12)
        })
    return pd.DataFrame(rows)

def cat_analysis(df: pd.DataFrame, col: str, top=15):
    vc = df[col].value_counts(dropna=False)
    keep = set(vc.head(top).index)
    d = df.copy()
    d[col] = d[col].where(d[col].isin(keep), other="__OTHER__")
    grp = d.groupby(col).apply(
        lambda g: pd.Series({
            "exposure": g["exposure"].sum(),
            "observed_rate": g["claim_count"].sum()/max(g["exposure"].sum(),1e-12),
            "pred_poi": np.average(g["rate_poisson"], weights=g["exposure"]),
            "pred_neb": np.average(g["rate_negbin"],  weights=g["exposure"]),
            "pred_gbm": np.average(g["rate_gbm"],     weights=g["exposure"]),
            "n": len(g),
        })
    ).reset_index().sort_values("exposure", ascending=False)
    return grp
def calibration_spectrum(df: pd.DataFrame, rate_col: str, n=20):
    """
    Równe wiadra po ekspozycji (ascending po predykcji).
    Zwraca: bin, exposure, exposure_share, observed_rate, pred_rate, n
    """
    d = df[["exposure","claim_count",rate_col]].copy().sort_values(rate_col, ascending=True)
    tot_e = d["exposure"].sum()
    cum_e = d["exposure"].cumsum()
    edges = [tot_e * i / n for i in range(1, n)]
    d["bin"] = np.searchsorted(edges, cum_e, side="right")

    out = (
        d.groupby("bin").apply(
            lambda g: pd.Series({
                "exposure": g["exposure"].sum(),
                "exposure_share": g["exposure"].sum()/max(tot_e,1e-12),
                "observed_rate": g["claim_count"].sum()/max(g["exposure"].sum(),1e-12),
                "pred_rate": np.average(g[rate_col], weights=g["exposure"]),
                "n": len(g)
            })
        ).reset_index().sort_values("bin")
    )
    return out

def num_analysis(df: pd.DataFrame, col: str, q=10):
    d = df.copy()
    try:
        d["bin"] = pd.qcut(d[col], q=q, duplicates="drop")
    except Exception:
        d["bin"] = pd.cut(d[col], bins=min(q, int(d[col].nunique())))
    grp = d.groupby("bin").apply(
        lambda g: pd.Series({
            "exposure": g["exposure"].sum(),
            "observed_rate": g["claim_count"].sum()/max(g["exposure"].sum(),1e-12),
            "pred_poi": np.average(g["rate_poisson"], weights=g["exposure"]),
            "pred_neb": np.average(g["rate_negbin"],  weights=g["exposure"]),
            "pred_gbm": np.average(g["rate_gbm"],     weights=g["exposure"]),
            "n": len(g),
            col+"_min": g[col].min(),
            col+"_max": g[col].max(),
        })
    ).reset_index().sort_values(col+"_min")
    return grp

def save_plot(fig, name):
    p = OUTD / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"[PNG] {p}")

# ---------- main ----------
def main():
    base = pd.read_parquet(PROC) if PROC.exists() else None
    preds = pd.read_parquet(PREDS)
    if "split" not in preds.columns:
        preds["split"] = "test"

    _ensure_cols(preds)

    df = preds.merge(
        base[["policy_id"] + CAT_COLS + NUM_COLS] if base is not None else preds[["policy_id"]],
        on="policy_id", how="left"
    )
    df = _add_rates(df)

    test = df[df["split"] == "test"].copy()

    # Outliery do CSV
    outliers = test[(test["exposure"] < 0.02) | (test["claim_count"] > 4)].copy()
    outliers.to_csv(OUTD / "outliers_test.csv", index=False)

    # METRYKI GLOBALNE
    metrics = []
    y_t = test["claim_count"].to_numpy(float)
    mu_poi = np.clip(test["yhat_poisson"].to_numpy(float), 1e-12, None)
    mu_neb = np.clip(test["yhat_negbin" ].to_numpy(float), 1e-12, None)
    mu_gbm = np.clip(test["yhat_gbm"    ].to_numpy(float), 1e-12, None)

    metrics.append(("poisson_deviance", "poisson", poisson_dev(y_t, mu_poi)))
    metrics.append(("poisson_deviance", "negbin",  poisson_dev(y_t, mu_neb)))
    metrics.append(("poisson_deviance", "gbm",     poisson_dev(y_t, mu_gbm)))
    met_df = pd.DataFrame(metrics, columns=["metric","model","value"])
    met_df.to_csv(OUTD / "metrics_test.csv", index=False)
    print("\n=== METRYKI (TEST) ===")
    print(met_df.pivot(index="metric", columns="model", values="value"))

    # Gini + Lorenz
    res_gini = []
    for m, col in MODEL_RATE_COLS.items():
        cx, cy = lorenz_points(
            y=test["claim_count"].to_numpy(float),
            w=test["exposure"].to_numpy(float),
            score=test[col].to_numpy(float),
            ascending=True
        )
        g = gini_from_lorenz(cx, cy)
        res_gini.append((m, g))
        fig = plt.figure()
        plt.plot(cx, cy, label=f"{m} (Gini={g:.3f})")
        plt.plot([0,1],[0,1], linestyle="--")
        plt.xlabel("Cumulative share of exposure")
        plt.ylabel("Cumulative share of claims")
        plt.title("Lorenz curve (test)")
        plt.legend()
        save_plot(fig, f"lorenz_{m}")
        gains = gains_table(test, col)
        gains.to_csv(OUTD / f"gains_{m}.csv", index=False)
    gini_df = pd.DataFrame(res_gini, columns=["model","gini"])
    gini_df.to_csv(OUTD / "gini_test.csv", index=False)

    # Kalibracja (decyle po ekspozycji)
    for m, col in MODEL_RATE_COLS.items():
        cal = decile_calibration(test, col, n=10)
        cal.to_csv(OUTD / f"calibration_{m}.csv", index=False)
        fig = plt.figure()
        plt.plot(cal["bin"], cal["observed_rate"], marker="o", label="observed")
        plt.plot(cal["bin"], cal["pred_rate"], marker="o", label=f"pred {m}")
        plt.xlabel("Decile (low → high risk)")
        plt.ylabel("Claim frequency")
        plt.title(f"Calibration by decile – {m}")
        plt.legend()
        save_plot(fig, f"calibration_{m}")

    # ===== SPEKTRUM KALIBRACJI (np. 20 wiader) + RELIABILITY =====
    spectrum_bins = 20
    for m, col in MODEL_RATE_COLS.items():
        spec = calibration_spectrum(test, col, n=spectrum_bins)
        spec.to_csv(OUTD / f"spectrum_{m}_n{spectrum_bins}.csv", index=False)

        # wykres: spektrum (observed vs predicted po binach)
        fig = plt.figure()
        x = np.arange(len(spec))
        plt.plot(x, spec["observed_rate"], marker="o", label="observed")
        plt.plot(x, spec["pred_rate"],    marker="o", label=f"pred {m}")
        plt.xlabel(f"Equal-exposure bins (n={spectrum_bins}) – low → high predicted risk")
        plt.ylabel("Claim frequency")
        plt.title(f"Calibration spectrum – {m}")
        plt.legend()
        save_plot(fig, f"spectrum_{m}_n{spectrum_bins}")

        # wykres: reliability (średnia pred w binie vs średnia obserwowana)
        fig = plt.figure()
        sizes = (2000 * spec["exposure_share"]).to_numpy()  # widoczne, ale nie agresywne
        plt.scatter(spec["pred_rate"], spec["observed_rate"], s=sizes, alpha=0.8)
        lim_min = 0.0
        lim_max = max(spec["pred_rate"].max(), spec["observed_rate"].max()) * 1.05
        plt.plot([lim_min, lim_max],[lim_min, lim_max], linestyle="--")
        plt.xlabel("Mean predicted rate (bin)")
        plt.ylabel("Observed rate (bin)")
        plt.title(f"Reliability plot – {m} (bins={spectrum_bins})")
        save_plot(fig, f"reliability_{m}_n{spectrum_bins}")

    # Rozkład predykcji (obcięcie 99p)
    fig = plt.figure()
    for m, col in MODEL_RATE_COLS.items():
        vals = test[col].to_numpy(float)
        mx = np.percentile(vals, 99)
        plt.hist(vals[vals <= mx], bins=60, density=True, alpha=0.5, label=m)
    plt.xlim(left=0)
    plt.xlabel("Predicted claim rate")
    plt.ylabel("Density")
    plt.title("Distribution of predicted rates (test)")
    plt.legend()
    save_plot(fig, "pred_rate_hist_test")

    # Analizy po zmiennych
    for c in CAT_COLS:
        if c in test.columns:
            tab = cat_analysis(test, c, top=15)
            tab.to_csv(OUTD / f"by_{c}.csv", index=False)
            head = tab.head(15)
            fig = plt.figure(figsize=(10,5))
            x = np.arange(len(head))
            plt.plot(x, head["observed_rate"], marker="o", label="observed")
            plt.plot(x, head["pred_poi"], marker="o", label="poisson")
            plt.plot(x, head["pred_neb"], marker="o", label="negbin")
            plt.plot(x, head["pred_gbm"], marker="o", label="gbm")
            plt.xticks(x, head[c].astype(str), rotation=60, ha="right")
            plt.ylabel("Claim frequency")
            plt.title(f"Observed vs predicted by {c} (top-15 by exposure) – test")
            plt.legend()
            save_plot(fig, f"by_{c}_plot")

    for c in NUM_COLS:
        if c in test.columns:
            tab = num_analysis(test, c, q=10)
            tab.to_csv(OUTD / f"by_{c}_bins.csv", index=False)
            fig = plt.figure()
            x = np.arange(len(tab))
            plt.plot(x, tab["observed_rate"], marker="o", label="observed")
            plt.plot(x, tab["pred_poi"], marker="o", label="poisson")
            plt.plot(x, tab["pred_neb"], marker="o", label="negbin")
            plt.plot(x, tab["pred_gbm"], marker="o", label="gbm")
            labels = [f"[{a:.0f},{b:.0f}]" for a,b in zip(tab[c+"_min"], tab[c+"_max"])]
            plt.xticks(x, labels, rotation=45, ha="right")
            plt.ylabel("Claim frequency")
            plt.title(f"Observed vs predicted by {c} bins – test")
            plt.legend()
            save_plot(fig, f"by_{c}_bins_plot")

    # Top-K capture + wykres
    rows = []
    for m, col in MODEL_RATE_COLS.items():
        t = topk_table(test, col)
        t["model"] = m
        rows.append(t)
    topk_all = pd.concat(rows, ignore_index=True)
    topk_all.to_csv(OUTD / "topk_capture.csv", index=False)

    fig = plt.figure()
    for m in MODEL_RATE_COLS:
        sub = topk_all[topk_all["model"] == m]
        plt.plot(sub["k_expo"], sub["claim_share"], marker="o", label=m)
    plt.plot([0,1],[0,1], linestyle="--", label="random")
    plt.xlabel("Top-k exposure share")
    plt.ylabel("Captured share of claims")
    plt.title("Gain curve (test)")
    plt.legend()
    save_plot(fig, "gain_curve")

    # Feature importance z GBM (jeśli model zapisany)
    gbm_pkl = Path("data/interim/models/freq_gbm.pkl")
    if gbm_pkl.exists():
        with open(gbm_pkl, "rb") as f:
            pack = pickle.load(f)
        model = pack["model"]; cols = pack.get("cols", [])
        imp = pd.DataFrame({"feature": cols, "gain": model.booster_.feature_importance(importance_type="gain")})
        imp = imp.sort_values("gain", ascending=False)
        imp.to_csv(OUTD / "gbm_importance.csv", index=False)

        head = imp.head(25)
        fig = plt.figure(figsize=(8,6))
        y = np.arange(len(head))
        plt.barh(y, head["gain"])
        plt.yticks(y, head["feature"])
        plt.gca().invert_yaxis()
        plt.xlabel("Gain")
        plt.title("LightGBM feature importance (top 25)")
        save_plot(fig, "gbm_importance_top25")

    print("\n[OK] Wyniki CSV/PNG w:", OUTD)

if __name__ == "__main__":
    main()
