from __future__ import annotations
from pathlib import Path
import io, base64, zipfile, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle

# ---- ścieżki ----
PROC = Path("data/processed/french_capped.parquet") if Path("data/processed/french_capped.parquet").exists() \
       else Path("data/processed/french.parquet")
PREDS = Path("data/interim/freq_preds.parquet")
MDIR  = Path("data/interim/models")
OUTD  = Path("data/interim/freq_analysis")
OUTD.mkdir(parents=True, exist_ok=True)

OUT_HTML = OUTD / "freq_model_report.html"
OUT_ZIP  = OUTD / "freq_model_report_bundle.zip"

# featury bazowe (jak w modelu)
NUM_COLS = ["driver_age","vehicle_age","vehicle_power","bonus_malus","density"]
CAT_COLS = ["vehicle_brand","fuel","area","region"]

# ---------- utils ----------
def poisson_dev(y, mu):
    mu = np.clip(np.asarray(mu, float), 1e-12, None)
    y  = np.asarray(y, float)
    # mean Poisson deviance (sklearn definicja)
    return float(np.mean(2 * (y * np.log(np.clip(y/mu, 1e-12, None)) - (y - mu))))

def lorenz_points(y, w, score, ascending=True):
    df = pd.DataFrame({"y": y, "w": w, "score": score}).copy()
    df = df.sort_values("score", ascending=ascending, kind="mergesort")
    wsum = df["w"].sum()
    ysum = df["y"].sum()
    cx = (df["w"].cumsum() / max(wsum, 1e-12)).to_numpy()
    cy = (df["y"].cumsum() / max(ysum, 1e-12)).to_numpy()
    cx = np.concatenate([[0.0], cx]); cy = np.concatenate([[0.0], cy])
    return cx, cy

def gini_from_lorenz(cx, cy):
    area = np.trapezoid(cy, cx)
    return float(1.0 - 2.0 * area)

def _img_b64(png_path: Path) -> str:
    if not png_path.exists(): return ""
    return "data:image/png;base64," + base64.b64encode(png_path.read_bytes()).decode("ascii")

def save_plot(fig, name):
    p = OUTD / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return p

def exposure_weighted_rate(df, rate_col=None):
    expo = df["exposure"].sum()
    if expo <= 0: return 0.0
    if rate_col is None:
        # observed
        return float(df["claim_count"].sum() / expo)
    return float(np.average(df[rate_col].to_numpy(float), weights=df["exposure"].to_numpy(float)))

def load_models():
    # Poisson
    pois = None
    p_pkl = MDIR / "freq_glm_poisson.pkl"
    if p_pkl.exists():
        try:
            pois = sm.iolib.summary.load(p_pkl)  # GLMResults
        except Exception:
            with open(p_pkl, "rb") as f: pois = pickle.load(f)
    # NegBin
    nb = None
    n_pkl = MDIR / "freq_glm_negbin.pkl"
    if n_pkl.exists():
        try:
            nb = sm.iolib.summary.load(n_pkl)
        except Exception:
            with open(n_pkl, "rb") as f: nb = pickle.load(f)
    # GBM
    gbm = None
    g_pkl = MDIR / "freq_gbm.pkl"
    if g_pkl.exists():
        with open(g_pkl, "rb") as f: gbm = pickle.load(f)  # dict(model, cols)
    return pois, nb, gbm

def glm_table(res) -> pd.DataFrame:
    # zwróć tabelę coef/std_err/z/pvalue
    try:
        summ = res.summary2().tables[1].reset_index()
        summ.columns = ["term","coef","std_err","z","pvalue","[0.025","0.975]"]
        return summ
    except Exception:
        # fallback
        params = pd.Series(getattr(res, "params", {}))
        bse    = pd.Series(getattr(res, "bse", {}))
        pval   = pd.Series(getattr(res, "pvalues", {}))
        out = pd.DataFrame({"term": params.index, "coef": params.values,
                            "std_err": bse.reindex(params.index).values,
                            "pvalue": pval.reindex(params.index).values})
        return out

def nb_alpha(res):
    # spróbuj wyciągnąć alpha z GLM-NB
    try:
        return float(res.model.family.alpha)
    except Exception:
        try:
            return float(res.family.alpha)
        except Exception:
            return None

def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in NUM_COLS:
        if c in df.columns:
            s = df[c]
            rows.append({
                "feature": c, "type": "numeric",
                "missing_rate": float(s.isna().mean()),
                "min": float(np.nanmin(s)), "p25": float(np.nanpercentile(s,25)),
                "mean": float(np.nanmean(s)), "p75": float(np.nanpercentile(s,75)),
                "max": float(np.nanmax(s)), "n_unique": int(s.nunique(dropna=True))
            })
    for c in CAT_COLS:
        if c in df.columns:
            s = df[c]
            rows.append({
                "feature": c, "type": "categorical",
                "missing_rate": float(s.isna().mean()),
                "n_unique": int(s.nunique(dropna=True)),
                "top": str(s.value_counts(dropna=False).head(3).to_dict())
            })
    return pd.DataFrame(rows)

RATIONALE = {
    "driver_age": "Wiek kierowcy koreluje z doświadczeniem/ryzykiem – skrajne grupy mają zwykle wyższą częstotliwość.",
    "vehicle_age": "Starsze pojazdy mogą mieć gorszy stan techniczny → inna częstość i koszt szkód.",
    "vehicle_power": "Moc/pojemność to proxy dla stylu jazdy i prędkości.",
    "bonus_malus": "Historia szkodowa w taryfie (BM) – silny predyktor future claims.",
    "density": "Gęstość zaludnienia ~ ekspozycja na ruch/miejskość ⇒ częstsze kolizje.",
    "vehicle_brand": "Marka/model grupują typ pojazdu i zachowania właścicieli.",
    "fuel": "Rodzaj paliwa różnicuje roczniki/segmenty i przebiegi.",
    "area": "Strefa taryfowa (miejskie/wiejskie).",
    "region": "Warunki drogowe, klimat, kultura jazdy, przepływy ruchu."
}

# ---------- główny raport ----------
def main():
    # wejścia
    if not PREDS.exists(): raise SystemExit("Brak data/interim/freq_preds.parquet – odpal najpierw trenowanie.")
    preds = pd.read_parquet(PREDS)
    if "split" not in preds.columns: preds["split"] = "test"
    base = pd.read_parquet(PROC) if PROC.exists() else None

    # modele
    pois_res, nb_res, gbm_pack = load_models()
    gbm_model = gbm_pack["model"] if isinstance(gbm_pack, dict) and "model" in gbm_pack else None

    # scal do analizy splitów
    df = preds.copy()
    # dodaj przewidywane often rates
    for col_pred, rate_col in [("yhat_poisson","rate_poisson"), ("yhat_negbin","rate_negbin"), ("yhat_gbm","rate_gbm")]:
        df[rate_col] = np.clip(df[col_pred].to_numpy(float) / np.clip(df["exposure"].to_numpy(float), 1e-12, None), 1e-12, None)

    splits = ["train","valid","test"]
    split_rows = []
    # metryki per split + krzywe Lorenza (overlay 3 modeli)
    lorenz_imgs = {}
    for sp in splits:
        part = df[df["split"]==sp].copy()
        if part.empty: continue
        obs_rate = exposure_weighted_rate(part, None)
        poi_rate = exposure_weighted_rate(part, "rate_poisson")
        neb_rate = exposure_weighted_rate(part, "rate_negbin")
        gbm_rate = exposure_weighted_rate(part, "rate_gbm")

        # deviance liczymy na liczbach szkód (mu = pred liczby szkód)
        y = part["claim_count"].to_numpy(float)
        expo = part["exposure"].to_numpy(float)
        mu_p = np.clip(part["yhat_poisson"].to_numpy(float), 1e-12, None)
        mu_n = np.clip(part["yhat_negbin" ].to_numpy(float), 1e-12, None)
        mu_g = np.clip(part["yhat_gbm"    ].to_numpy(float), 1e-12, None)
        dev_p = poisson_dev(y, mu_p); dev_n = poisson_dev(y, mu_n); dev_g = poisson_dev(y, mu_g)

        # Gini (po ekspozycji, sort po rate rosnąco)
        cx_p, cy_p = lorenz_points(y, expo, part["rate_poisson"].to_numpy(float), ascending=True)
        cx_n, cy_n = lorenz_points(y, expo, part["rate_negbin" ].to_numpy(float), ascending=True)
        cx_g, cy_g = lorenz_points(y, expo, part["rate_gbm"    ].to_numpy(float), ascending=True)
        g_p = gini_from_lorenz(cx_p, cy_p); g_n = gini_from_lorenz(cx_n, cy_n); g_g = gini_from_lorenz(cx_g, cy_g)

        split_rows.append({
            "split": sp,
            "N": int(len(part)),
            "exposure_sum": float(expo.sum()),
            "observed_rate": obs_rate,
            "pred_rate_poisson": poi_rate,
            "pred_rate_negbin":  neb_rate,
            "pred_rate_gbm":     gbm_rate,
            "poisson_dev_poisson": dev_p,
            "poisson_dev_negbin":  dev_n,
            "poisson_dev_gbm":     dev_g,
            "gini_poisson": g_p,
            "gini_negbin":  g_n,
            "gini_gbm":     g_g
        })

        # rysunek: krzywe Lorenza – overlay 3 modele
        fig = plt.figure()
        plt.plot(cx_g, cy_g, label=f"GBM (Gini={g_g:.3f})", linestyle=":", linewidth=2.0, alpha=0.95, zorder=2)
        plt.plot(cx_n, cy_n, label=f"NegBin (Gini={g_n:.3f})", linestyle="--", linewidth=1.8, alpha=0.95, zorder=3)
        plt.plot(cx_p, cy_p, label=f"Poisson (Gini={g_p:.3f})", linestyle="-", linewidth=2.6, alpha=1.00, zorder=4)
        plt.plot([0, 1], [0, 1], linestyle="-.", linewidth=1.0, alpha=0.6, zorder=1)
        plt.xlabel("Cumulative exposure");
        plt.ylabel("Cumulative claims")
        plt.title(f"Lorenz curves – {sp}")
        plt.legend()
        png = save_plot(fig, f"lorenz_{sp}_overlay")
        lorenz_imgs[sp] = _img_b64(png)

    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(OUTD / "splits_summary.csv", index=False)

    # -------- GLM tabele (coef/p-value) + formuły ----------
    glm_sections = []
    if pois_res is not None:
        pois_tab = glm_table(pois_res)
        pois_tab.to_csv(OUTD / "glm_poisson_coeffs.csv", index=False)
        glm_sections.append(("Poisson",
            "log(mu) = β0 + β·X + log(exposure); link: log; rodzina: Poisson",
            pois_tab))
    if nb_res is not None:
        nb_tab = glm_table(nb_res)
        nb_tab.to_csv(OUTD / "glm_negbin_coeffs.csv", index=False)
        alpha = nb_alpha(nb_res)
        descr = f"log(mu) = β0 + β·X + log(exposure); link: log; rodzina: Negative Binomial (alpha≈{alpha:.3g})" if alpha is not None \
                else "log(mu) = β0 + β·X + log(exposure); link: log; rodzina: Negative Binomial"
        glm_sections.append(("Negative Binomial", descr, nb_tab))

    # -------- Parametry LightGBM ----------
    gbm_params = None; gbm_best_iter = None; gbm_best_score = None
    if gbm_model is not None:
        try:
            gbm_params = gbm_model.get_params()
        except Exception:
            gbm_params = {}
        gbm_best_iter = getattr(gbm_model, "best_iteration_", None)
        gbm_best_score = getattr(gbm_model, "best_score_", None)
        # zapisz paramy
        with open(OUTD / "gbm_params.json", "w", encoding="utf-8") as f:
            json.dump({
                "params": gbm_params,
                "best_iteration_": gbm_best_iter,
                "best_score_": gbm_best_score
            }, f, ensure_ascii=False, indent=2)

    # -------- Przegląd zmiennych + uzasadnienia ----------
    var_summary = None
    if base is not None:
        var_summary = feature_summary(base)
        var_summary.to_csv(OUTD / "variables_summary.csv", index=False)

    # -------- HTML raport ----------
    css = """
    <style>
    body { font:14px/1.5 system-ui, sans-serif; margin:24px; }
    h1,h2,h3 { margin-top:24px; }
    table { border-collapse: collapse; margin: 10px 0; width:100%; }
    th, td { padding: 6px 10px; border-bottom: 1px solid #e5e5e5; text-align:right; }
    th:first-child, td:first-child { text-align:left; }
    code { background:#f6f8fa; padding:2px 4px; border-radius:4px; }
    .muted { color:#666; font-size:12px; }
    .imgbox { border:1px solid #eee; padding:10px; margin:10px 0; }
    .grid { display:grid; gap:16px; grid-template-columns: repeat(auto-fit, minmax(320px,1fr)); }
    </style>
    """
    html = io.StringIO()
    html.write("<!doctype html><html><head><meta charset='utf-8'><title>Frequency – detailed model report</title>")
    html.write(css)
    html.write("</head><body>")
    html.write("<h1>Frequency – detailed model report</h1>")

    # Inputs
    html.write("<h2>Inputs</h2><ul>")
    html.write(f"<li>Predictions: <code>{PREDS.as_posix()}</code> (N={len(preds):,})</li>")
    if PROC.exists():
        html.write(f"<li>Base dataset: <code>{PROC.as_posix()}</code></li>")
    html.write(f"<li>Models dir: <code>{MDIR.as_posix()}</code></li>")
    html.write("</ul>")
    html.write("<p class='muted'>Split: 70% train / 15% valid / 15% test, stratyfikacja po has_claim; offset = log(exposure); capping: ClaimNb≤4, Exposure≤1 oraz usunięte Exposure&lt;0.02.</p>")

    # Split summary
    html.write("<h2>Train/Valid/Test – częstość i metryki</h2>")
    if not split_df.empty:
        html.write(split_df.to_html(index=False, border=0))
    else:
        html.write("<p class='muted'>Brak danych o splitach.</p>")

    # Lorenz curves
    html.write("<h2>Lorenz curves (overlay 3 models)</h2><div class='grid'>")
    for sp in splits:
        b64 = lorenz_imgs.get(sp, "")
        if b64:
            html.write(f"<div class='imgbox'><div><b>{sp}</b></div><img src='{b64}' style='max-width:100%'></div>")
    html.write("</div>")

    # GLM rows
    html.write("<h2>GLM models (formuła i istotność)</h2>")
    for name, desc, table in glm_sections:
        html.write(f"<h3>{name}</h3>")
        html.write(f"<p><code>{desc}</code></p>")
        # pokaż top 30 wg p-value
        disp = table.copy()
        if "pvalue" in disp.columns:
            def _fmt(p):
                try:
                    p = float(p)
                except Exception:
                    return ""
                return "<0.01" if p < 0.01 else f"{p:.2f}"

            disp["pvalue"] = disp["pvalue"].apply(_fmt)
        html.write(disp.head(30).to_html(index=False, border=0))

    # LightGBM params
    html.write("<h2>LightGBM – parametry i best iteration</h2>")
    if gbm_params is not None:
        html.write("<pre>"+json.dumps({
            "params": gbm_params,
            "best_iteration_": gbm_best_iter,
            "best_score_": gbm_best_score
        }, ensure_ascii=False, indent=2)+"</pre>")
    else:
        html.write("<p class='muted'>Model GBM nie został znaleziony.</p>")

    # Variables summary + rationale
    html.write("<h2>Variables – summary & rationale</h2>")
    if var_summary is not None and not var_summary.empty:
        html.write(var_summary.to_html(index=False, border=0))
        html.write("<h3>Rationale (business justification)</h3><ul>")
        for k in NUM_COLS + CAT_COLS:
            if k in (base.columns if base is not None else []):
                html.write(f"<li><b>{k}</b>: {RATIONALE.get(k, '—')}</li>")
        html.write("</ul>")
    else:
        html.write("<p class='muted'>Brak bazy do przeglądu zmiennych.</p>")

    # Production checklist
    html.write("<h2>Production checklist (do odhaczenia)</h2><ul>")
    html.write("<li>Stabilność metryk: monitoring Poisson deviance i Gini w czasie (per segment).</li>")
    html.write("<li>Kalibracja: tabela decylowa po ekspozycji dla nowych okresów; brak driftu.</li>")
    html.write("<li>Wejścia: walidacja schematu (typy, zakresy, słowniki kategorii, brakujące wartości).</li>")
    html.write("<li>Explainability: SHAP/feature importance dla GBM, współczynniki GLM + monotoniczność oczekiwana.</li>")
    html.write("<li>Retrain polityka: częstotliwość, windowing, wersjonowanie danych i modeli (pickle + hash danych).</li>")
    html.write("<li>Repro: pinned deps (requirements/lock), stały seed, zapis splitu i listy kolumn (dummies).</li>")
    html.write("<li>Ryzyko regulacyjne: tylko zmienne dozwolone, bez proxy cech wrażliwych; uzasadnienia (powyżej).</li>")
    html.write("<li>Backtesting: stabilność w miesiącach/rocznikach, testy przeciążeniowe (ekstrema featurów).</li>")
    html.write("</ul>")

    html.write("<p class='muted'>Auto-generated detailed report. Files placed in data/interim/freq_analysis/</p>")
    html.write("</body></html>")
    OUT_HTML.write_text(html.getvalue(), encoding="utf-8")
    print(f"[OK] HTML -> {OUT_HTML}")

    # ZIP bundle (HTML + CSV + obrazy)
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(OUT_HTML, arcname="freq_model_report.html")
        extras = [
            "splits_summary.csv", "glm_poisson_coeffs.csv", "glm_negbin_coeffs.csv",
            "variables_summary.csv", "gbm_params.json",
            "lorenz_train_overlay.png", "lorenz_valid_overlay.png", "lorenz_test_overlay.png"
        ]
        for e in extras:
            p = OUTD / e
            if p.exists(): z.write(p, arcname=f"analysis/{e}")
    print(f"[OK] ZIP  -> {OUT_ZIP}")

if __name__ == "__main__":
    main()
