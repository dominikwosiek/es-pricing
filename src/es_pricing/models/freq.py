from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_poisson_deviance
import json

# ===== ścieżki =====
DATA = Path("data/processed/french_capped.parquet") if Path("data/processed/french_capped.parquet").exists() \
       else Path("data/processed/french.parquet")
OUTD = Path("data/interim"); OUTD.mkdir(parents=True, exist_ok=True)
MDIR = OUTD / "models"; MDIR.mkdir(parents=True, exist_ok=True)

# ===== featury =====
NUM_COLS = ["driver_age","vehicle_age","vehicle_power","bonus_malus","density"]
CAT_COLS = ["vehicle_brand","fuel","area","region"]
# ---- GLM: binnings + design ----
GLM_BIN_NUMS = ["driver_age","vehicle_age","vehicle_power","density"]
GLM_LINEAR_NUMS = ["bonus_malus"]

def _fit_bins(train: pd.DataFrame, q=10):
    bins = {}
    for col in GLM_BIN_NUMS:
        if col in train.columns:
            # kwantyle 0..100%, usunięcie duplikatów
            qs = np.linspace(0, 1, q+1)
            edges = np.unique(np.quantile(train[col].to_numpy(float), qs))
            if len(edges) < 3:
                # awaryjnie 3 przedziały
                vmin, vmax = np.nanmin(train[col]), np.nanmax(train[col])
                edges = np.unique([vmin, (vmin+vmax)/2, vmax])
            bins[col] = edges.tolist()
    return bins

def _apply_bins(df: pd.DataFrame, bins: dict) -> pd.DataFrame:
    parts = []
    # dummies z binów dla wybranych numerycznych
    for col, edges in bins.items():
        if col in df.columns:
            cats = pd.cut(df[col], bins=np.array(edges), include_lowest=True, duplicates="drop")
            d = pd.get_dummies(cats, prefix=col, drop_first=True)
            parts.append(d)
    # linear dla bonus_malus (jeśli jest)
    lin = pd.DataFrame(index=df.index)
    for col in GLM_LINEAR_NUMS:
        if col in df.columns:
            lin[col] = pd.to_numeric(df[col], errors="coerce")
    # kategorie klasycznie
    cats = pd.get_dummies(df[[c for c in CAT_COLS if c in df.columns]], drop_first=True)
    X = pd.concat([lin, *parts, cats], axis=1).astype(float).fillna(0.0)
    return X

def build_glm_mats(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    bins = _fit_bins(train, q=10)
    Xtr = _apply_bins(train, bins)
    Xva = _apply_bins(valid, bins).reindex(columns=Xtr.columns, fill_value=0.0)
    Xte = _apply_bins(test,  bins).reindex(columns=Xtr.columns, fill_value=0.0)
    return Xtr, Xva, Xte, Xtr.columns.tolist(), bins

def backward_elimination_poisson(X, y, exposure, protected=("bonus_malus",), p_thresh=0.05, max_iter=20):
    cols = list(X.columns)
    for _ in range(max_iter):
        Xc = sm.add_constant(X[cols], has_constant="add")
        res = sm.GLM(y, Xc, family=sm.families.Poisson(), offset=np.log(exposure)).fit(cov_type="HC0")
        # tabela p-value
        summ = res.summary2().tables[1]
        # nazwy bez const
        pv = summ["P>|z|"].drop(index="const", errors="ignore")
        # nie ruszaj chronionych
        pv = pv[~pv.index.to_series().str.startswith(protected)]
        worst = pv.idxmax() if not pv.empty else None
        if worst is None or pv.max() <= p_thresh:
            return res, cols  # koniec
        # usuń najgorszy
        cols.remove(worst)
    # limit iteracji – zwróć ostatni fit
    Xc = sm.add_constant(X[cols], has_constant="add")
    res = sm.GLM(y, Xc, family=sm.families.Poisson(), offset=np.log(exposure)).fit(cov_type="HC0")
    return res, cols

def build_matrix(df: pd.DataFrame):
    y = pd.to_numeric(df["claim_count"], errors="coerce").fillna(0).astype(float).values
    exposure = pd.to_numeric(df["exposure"], errors="coerce").fillna(0).astype(float).values
    use_cols = [c for c in NUM_COLS if c in df.columns] + [c for c in CAT_COLS if c in df.columns]
    X_all = pd.get_dummies(df[use_cols], drop_first=True)
    X_all = X_all.replace([np.inf, -np.inf], np.nan).astype(float).fillna(0.0)
    return X_all, y, exposure, X_all.columns.tolist()

def align_matrix(df: pd.DataFrame, cols):
    use_cols = [c for c in NUM_COLS if c in df.columns] + [c for c in CAT_COLS if c in df.columns]
    X = pd.get_dummies(df[use_cols], drop_first=True)
    X = X.reindex(columns=cols, fill_value=0.0)
    X = X.replace([np.inf, -np.inf], np.nan).astype(float).fillna(0.0)
    return X

def fit_poisson(X, y, exposure):
    Xp = sm.add_constant(X.astype(float), has_constant="add")
    model = sm.GLM(y, Xp, family=sm.families.Poisson(), offset=np.log(exposure))
    res = model.fit(cov_type="HC0")  # robust SE
    return res

def fit_negbin(X, y, exposure, X_val, y_val, ex_val):
    Xp = sm.add_constant(X.astype(float), has_constant="add")
    Xv = sm.add_constant(X_val.astype(float), has_constant="add")
    best = None
    best_dev = np.inf
    for alpha in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
        fam = sm.families.NegativeBinomial(alpha=alpha)
        try:
            res = sm.GLM(y, Xp, family=fam, offset=np.log(exposure)).fit()
            mu_val = np.clip(res.predict(Xv, offset=np.log(ex_val)), 1e-12, None)
            dev = mean_poisson_deviance(y_val, mu_val)
            if dev < best_dev:
                best, best_dev = (res, alpha), dev
        except Exception:
            continue
    if best is None:
        # fallback na Poissona
        fam = sm.families.Poisson()
        best = (sm.GLM(y, Xp, family=fam, offset=np.log(exposure)).fit(), None)
    return best[0]

def fit_gbm_rate(X_tr, y_tr, ex_tr, X_val, y_val, ex_val):
    rate_tr = y_tr / np.clip(ex_tr, 1e-9, None)
    rate_val = y_val / np.clip(ex_val, 1e-9, None)

    candidates = [
        dict(n_estimators=1200,learning_rate=0.02,num_leaves=127,min_data_in_leaf=600,  lambda_l2=5.0,  max_bin=511),
        dict(n_estimators=2000,learning_rate=0.02,num_leaves=255,min_data_in_leaf=800,  lambda_l2=10.0, max_bin=511),
        dict(n_estimators=4000,learning_rate=0.015,num_leaves=255,min_data_in_leaf=1000,lambda_l2=20.0, max_bin=511),
    ]
    best = None; best_dev = np.inf

    for params in candidates:
        gbm = LGBMRegressor(
            objective="poisson",
            subsample=0.8, colsample_bytree=0.8, bagging_freq=1,
            random_state=42, **params
        )
        gbm.fit(
            X_tr, rate_tr,
            sample_weight=ex_tr,
            eval_set=[(X_val, rate_val)],
            eval_sample_weight=[ex_val],
            eval_metric="poisson",
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        mu_val = np.clip(gbm.predict(X_val), 1e-12, None) * ex_val
        dev = mean_poisson_deviance(y_val, mu_val)
        if dev < best_dev:
            best, best_dev = gbm, dev
    return best


def predict_all(res_pois, res_nb, gbm_pack, frame: pd.DataFrame, glm_cols, glm_bins, tree_cols):
    # GLM
    X_glm = _apply_bins(frame, glm_bins).reindex(columns=glm_cols, fill_value=0.0)
    Xc = sm.add_constant(X_glm, has_constant="add")
    expo = frame["exposure"].to_numpy(float)
    mu_p = np.clip(res_pois.predict(Xc, offset=np.log(expo)), 1e-12, None)
    mu_n = np.clip(res_nb.predict(Xc,  offset=np.log(expo)), 1e-12, None)
    # GBM
    X_tree = align_matrix(frame, tree_cols)
    rate_g = np.clip(gbm_pack["model"].predict(X_tree), 1e-12, None)
    mu_g  = rate_g * expo
    return pd.DataFrame({
        "policy_id": frame["policy_id"].values,
        "exposure": expo,
        "claim_count": frame["claim_count"].astype(float).values,
        "yhat_poisson": mu_p,
        "yhat_negbin":  mu_n,
        "yhat_gbm":     mu_g,
        "split": frame.get("split","test")
    })



def main():
    df = pd.read_parquet(DATA)
    df = df[df["exposure"] > 0].copy()
    df["has_claim"] = (df["claim_count"] > 0).astype(int)

    # train/valid/test: 70/15/15, stratyfikacja
    train_full, test = train_test_split(df, test_size=0.15, random_state=42, stratify=df["has_claim"])
    train, valid = train_test_split(train_full, test_size=0.1765, random_state=42, stratify=train_full["has_claim"])  # 0.1765 ~ 15% całości

    for part in (train, valid, test):
        part.drop(columns=["has_claim"], inplace=True, errors="ignore")

    # --- macierze dla drzew (GBM) ---
    Xtr_tree, ytr, extr, tree_cols = build_matrix(train)
    Xva_tree = align_matrix(valid, tree_cols);
    yva = valid["claim_count"].to_numpy(float);
    exva = valid["exposure"].to_numpy(float)
    Xte_tree = align_matrix(test, tree_cols);
    yte = test["claim_count"].to_numpy(float);
    exte = test["exposure"].to_numpy(float)

    # --- macierze dla GLM (binnings kwantylowe + kategorie + bonus_malus liniowo) ---
    Xtr_glm, Xva_glm, Xte_glm, glm_cols, glm_bins = build_glm_mats(train, valid, test)

    # ===== Poisson (GLM) z eliminacją wsteczną =====
    print("[Poisson] fitting (GLM + backward elimination)...")
    pois_res, glm_cols_final = backward_elimination_poisson(Xtr_glm, ytr, extr, protected=("bonus_malus",),
                                                            p_thresh=0.05)
    # dopasuj walidację/test po redukcji kolumn
    Xva_glm_f = sm.add_constant(Xva_glm.reindex(columns=glm_cols_final, fill_value=0.0), has_constant="add")
    Xte_glm_f = sm.add_constant(Xte_glm.reindex(columns=glm_cols_final, fill_value=0.0), has_constant="add")
    print(pois_res.summary().as_text().splitlines()[:12])

    # ===== NegBin (GLM-NB, alpha grid) na tych samych kolumnach =====
    print("[NegBin ] fitting (GLM NB, alpha grid, reduced cols)...")

    def _fit_nb_on_cols(cols):
        Xp_tr = sm.add_constant(Xtr_glm.reindex(columns=cols, fill_value=0.0), has_constant="add")
        Xp_va = sm.add_constant(Xva_glm.reindex(columns=cols, fill_value=0.0), has_constant="add")
        best = None;
        best_dev = np.inf
        for alpha in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
            fam = sm.families.NegativeBinomial(alpha=alpha)
            try:
                res = sm.GLM(ytr, Xp_tr, family=fam, offset=np.log(extr)).fit()
                mu_va = np.clip(res.predict(Xp_va, offset=np.log(exva)), 1e-12, None)
                dev = mean_poisson_deviance(yva, mu_va)
                if dev < best_dev:
                    best, best_dev = res, dev
            except Exception:
                continue
        return best

    nb_res = _fit_nb_on_cols(glm_cols_final)
    try:
        print(nb_res.summary().as_text().splitlines()[:12])
    except Exception:
        print("[NegBin ] summary unavailable (ok).")

    # ===== GBM =====
    print("[GBM    ] fitting (early stopping + stronger grid)...")
    gbm = fit_gbm_rate(Xtr_tree, ytr, extr, Xva_tree, yva, exva)

    # ===== walidacja/dev =====
    def dev(y, mu): return mean_poisson_deviance(y, np.clip(mu, 1e-12, None))

    mu_p_tr = pois_res.predict(sm.add_constant(Xtr, has_constant="add"), offset=np.log(extr))
    mu_p_va = pois_res.predict(sm.add_constant(Xva, has_constant="add"), offset=np.log(exva))
    mu_p_te = pois_res.predict(sm.add_constant(Xte, has_constant="add"), offset=np.log(exte))

    mu_n_tr = nb_res.predict(sm.add_constant(Xtr, has_constant="add"), offset=np.log(extr))
    mu_n_va = nb_res.predict(sm.add_constant(Xva, has_constant="add"), offset=np.log(exva))
    mu_n_te = nb_res.predict(sm.add_constant(Xte, has_constant="add"), offset=np.log(exte))

    rate_g_tr = gbm.predict(Xtr); mu_g_tr = rate_g_tr * extr
    rate_g_va = gbm.predict(Xva); mu_g_va = rate_g_va * exva
    rate_g_te = gbm.predict(Xte); mu_g_te = rate_g_te * exte

    print("\n=== Poisson deviance (niższy lepszy) ===")
    print(f"Train: Poisson={dev(ytr, mu_p_tr):.6f}  NegBin={dev(ytr, mu_n_tr):.6f}  GBM={dev(ytr, mu_g_tr):.6f}")
    print(f"Valid: Poisson={dev(yva, mu_p_va):.6f}  NegBin={dev(yva, mu_n_va):.6f}  GBM={dev(yva, mu_g_va):.6f}")
    print(f"Test : Poisson={dev(yte, mu_p_te):.6f}  NegBin={dev(yte, mu_n_te):.6f}  GBM={dev(yte, mu_g_te):.6f}")

    # ===== zapisz modele =====
    pois_res.save(str(MDIR / "freq_glm_poisson.pkl"))
    try:
        nb_res.save(str(MDIR / "freq_glm_negbin.pkl"))
    except Exception:
        with open(MDIR / "freq_glm_negbin.pkl", "wb") as f:
            pickle.dump(nb_res, f)
    with open(MDIR / "freq_gbm.pkl", "wb") as f:
        pickle.dump({"model": gbm, "cols": cols}, f)

    # ===== predykcje dla wszystkich (z etykietą split) =====
    preds_tr = predict_all(pois_res, nb_res, {"model": gbm}, train.assign(split="train"), glm_cols_final, glm_bins,
                           tree_cols)
    preds_va = predict_all(pois_res, nb_res, {"model": gbm}, valid.assign(split="valid"), glm_cols_final, glm_bins,
                           tree_cols)
    preds_te = predict_all(pois_res, nb_res, {"model": gbm}, test.assign(split="test"), glm_cols_final, glm_bins,
                           tree_cols)
    with open(MDIR / "freq_glm_design.json", "w", encoding="utf-8") as f:
        json.dump({"glm_cols": glm_cols_final, "glm_bins": glm_bins}, f, ensure_ascii=False, indent=2)

    preds = pd.concat([preds_tr, preds_va, preds_te], ignore_index=True)
    outp = OUTD / "freq_preds.parquet"
    preds.to_parquet(outp, index=False)
    print(f"\n[OK] predictions -> {outp}")

if __name__ == "__main__":
    main()
