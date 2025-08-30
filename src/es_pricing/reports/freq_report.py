from __future__ import annotations
from pathlib import Path
import base64, io, zipfile
import pandas as pd

PROC = Path("data/processed/french.parquet")
PRED = Path("data/interim/freq_preds.parquet")
ADIR = Path("data/interim/freq_analysis")
RAW  = Path("data/raw")

OUT_HTML = ADIR / "freq_report.html"
OUT_ZIP  = ADIR / "freq_report_bundle.zip"

IMG_NAMES = [
    "lorenz_poisson.png", "lorenz_negbin.png", "lorenz_gbm.png",
    "calibration_poisson.png", "calibration_negbin.png", "calibration_gbm.png",
    "pred_rate_hist_test.png", "gain_curve.png",
    "gbm_importance_top25.png", "spectrum_poisson_n20.png", "spectrum_negbin_n20.png", "spectrum_gbm_n20.png",
    "reliability_poisson_n20.png", "reliability_negbin_n20.png", "reliability_gbm_n20.png",

]
CAT_COLS = ["vehicle_brand", "fuel", "area", "region"]
NUM_COLS = ["driver_age", "vehicle_age", "vehicle_power", "bonus_malus", "density"]

def _img_b64(p: Path) -> str:
    if not p.exists():
        return ""
    data = p.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")

def _tbl_html(df: pd.DataFrame, max_rows=20) -> str:
    df2 = df.head(max_rows)
    return df2.to_html(index=False, border=0)

def main():
    ADIR.mkdir(parents=True, exist_ok=True)

    if not PRED.exists():
        raise SystemExit("Brak data/interim/freq_preds.parquet. Najpierw odpal models/freq.py")
    preds = pd.read_parquet(PRED)
    if "split" not in preds.columns:
        preds["split"] = "test"

    base = pd.read_parquet(PROC) if PROC.exists() else None

    metrics = (ADIR / "metrics_test.csv")
    gini    = (ADIR / "gini_test.csv")
    metrics = pd.read_csv(metrics) if metrics.exists() else None
    gini    = pd.read_csv(gini)    if gini.exists()    else None

    cal_files = {m: ADIR / f"calibration_{m}.csv" for m in ("poisson","negbin","gbm")}
    cals = {m: (pd.read_csv(p) if p.exists() else None) for m,p in cal_files.items()}

    gains = {}
    for m in ("poisson","negbin","gbm"):
        p = ADIR / f"gains_{m}.csv"
        gains[m] = pd.read_csv(p) if p.exists() else None

    topk = ADIR / "topk_capture.csv"
    topk = pd.read_csv(topk) if topk.exists() else None

    imgs = {name: _img_b64(ADIR/name) for name in IMG_NAMES}

    test = preds[preds["split"]=="test"].copy()
    head_base = base.head(10) if base is not None else pd.DataFrame()
    raw_list = [p for p in [RAW/"freMTPL2freq.csv", RAW/"freMTPL2sev.csv"] if p.exists()]

    css = """
    <style>
    body { font: 14px/1.4 system-ui, sans-serif; margin: 24px; }
    h1,h2 { margin-top: 28px; }
    table { border-collapse: collapse; margin: 10px 0; }
    th, td { padding: 6px 10px; border-bottom: 1px solid #ddd; }
    .grid { display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
    .imgbox { border:1px solid #eee; padding:10px; }
    .muted { color:#666; font-size:12px; }
    </style>
    """
    html = io.StringIO()
    html.write("<!doctype html><html><head><meta charset='utf-8'><title>Frequency report</title>")
    html.write(css)
    html.write("</head><body>")
    html.write("<h1>Frequency – model comparison report</h1>")

    html.write("<h2>Inputs</h2>")
    html.write("<ul>")
    html.write(f"<li>Predictions: <code>{PRED.as_posix()}</code> (N={len(preds):,})</li>")
    if base is not None:
        html.write(f"<li>Base dataset: <code>{PROC.as_posix()}</code> (N={len(base):,})</li>")
    for rp in raw_list:
        html.write(f"<li>Raw: <code>{rp.as_posix()}</code> ({rp.stat().st_size//1024} KB)</li>")
    html.write("</ul>")

    html.write("<h2>Global metrics (test)</h2>")
    if metrics is not None:
        pivot = metrics.pivot(index="metric", columns="model", values="value").reset_index()
        html.write(_tbl_html(pivot))
        html.write("<p class='muted'>Note: preprocessing uses caps (ClaimNb≤4, Exposure≤1) and drops Exposure&lt;0.02 for stability.</p>")
    else:
        html.write("<p class='muted'>Brak metrics_test.csv – uruchom analysis/freq_analysis.py</p>")

    if gini is not None:
        html.write("<h3>Gini (test)</h3>")
        html.write(_tbl_html(gini))

    html.write("<div class='grid'>")
    for name in ("lorenz_poisson.png","lorenz_negbin.png","lorenz_gbm.png"):
        if imgs.get(name):
            html.write(f"<div class='imgbox'><img src='{imgs[name]}' style='max-width:100%'><div class='muted'>{name}</div></div>")
    html.write("</div>")

    html.write("<h2>Calibration by decile (test)</h2>")
    for m in ("poisson","negbin","gbm"):
        c = cals.get(m)
        if c is not None:
            html.write(f"<h3>{m}</h3>")
            html.write(_tbl_html(c))
            name = f"calibration_{m}.png"
            if imgs.get(name):
                html.write(f"<div class='imgbox'><img src='{imgs[name]}' style='max-width:100%'></div>")

    if imgs.get("pred_rate_hist_test.png"):
        html.write("<h2>Predicted rate distribution (test)</h2>")
        html.write(f"<div class='imgbox'><img src='{imgs['pred_rate_hist_test.png']}' style='max-width:100%'></div>")
    if imgs.get("gain_curve.png"):
        html.write("<h2>Gain curve (test)</h2>")
        html.write(f"<div class='imgbox'><img src='{imgs['gain_curve.png']}' style='max-width:100%'></div>")

    html.write("<h2>Gains tables</h2>")
    for m,tbl in gains.items():
        if tbl is not None:
            html.write(f"<h3>{m}</h3>")
            html.write(_tbl_html(tbl))

    html.write("<h2>Data snapshots</h2>")
    if base is not None and not head_base.empty:
        html.write("<h3>Base (first 10 rows)</h3>")
        html.write(_tbl_html(head_base, 10))
    html.write("<h3>Predictions (test sample)</h3>")
    html.write(_tbl_html(test.sample(min(1000,len(test)), random_state=1), 20))

    # Calibration spectrum + Reliability
    html.write("<h2>Calibration spectrum & Reliability (test)</h2>")
    html.write("<p class='muted'>Bins: equal exposure (n=20). Sprawdź skrajne koszyki: czy obserwowana ≈ przewidywana?</p>")
    # w siatce: po 2 obrazki na model
    html.write("<div class='grid'>")
    for m in ("poisson","negbin","gbm"):
        sp = f"spectrum_{m}_n20.png"
        rl = f"reliability_{m}_n20.png"
        if imgs.get(sp):
            html.write(f"<div class='imgbox'><b>{m} – spectrum</b><br><img src='{imgs[sp]}' style='max-width:100%'></div>")
        if imgs.get(rl):
            html.write(f"<div class='imgbox'><b>{m} – reliability</b><br><img src='{imgs[rl]}' style='max-width:100%'></div>")
    html.write("</div>")

    # skrajne koszyki – szybka tabela (pierwszy i ostatni bin)
    html.write("<h3>Extreme bins snapshot</h3>")
    for m in ("poisson","negbin","gbm"):
        p = ADIR / f"spectrum_{m}_n20.csv"
        if p.exists():
            t = pd.read_csv(p)
            ext = pd.concat([t.head(1), t.tail(1)], ignore_index=True)
            ext.loc[0,"_which"] = "lowest pred"
            ext.loc[1,"_which"] = "highest pred"
            cols = ["_which","exposure","exposure_share","observed_rate","pred_rate","n"]
            html.write(f"<h4>{m}</h4>")
            html.write(ext[cols].to_html(index=False, border=0))

    # GBM importances
    imp_csv = ADIR / "gbm_importance.csv"
    if imp_csv.exists():
        html.write("<h2>GBM feature importance</h2>")
        imp = pd.read_csv(imp_csv)
        html.write(_tbl_html(imp.head(30)))
        if imgs.get("gbm_importance_top25.png"):
            html.write(f"<div class='imgbox'><img src='{imgs['gbm_importance_top25.png']}' style='max-width:100%'></div>")

    html.write("<p class='muted'>Auto-generated report. Files placed in data/interim/freq_analysis/</p>")
    html.write("</body></html>")

    OUT_HTML.write_text(html.getvalue(), encoding="utf-8")
    print(f"[OK] HTML -> {OUT_HTML}")

    # ZIP bundle
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(OUT_HTML, arcname="freq_report.html")
        for rp in raw_list:
            z.write(rp, arcname=f"raw/{rp.name}")
        for extra in ["metrics_test.csv","gini_test.csv","topk_capture.csv",
                      "gains_poisson.csv","gains_negbin.csv","gains_gbm.csv",
                      "gbm_importance.csv"]:
            p = ADIR/extra
            if p.exists():
                z.write(p, arcname=f"analysis/{extra}")
        # dorzuć obrazki
        for name in IMG_NAMES:
            p = ADIR / name
            if p.exists():
                z.write(p, arcname=f"plots/{name}")
    print(f"[OK] ZIP  -> {OUT_ZIP}")

if __name__ == "__main__":
    main()
