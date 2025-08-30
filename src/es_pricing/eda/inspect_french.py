from pathlib import Path
import re
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/interim")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def pick_main_csv() -> Path:
    files = sorted(RAW_DIR.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise SystemExit("Brak CSV w data/raw/. Najpierw pobierz dane.")
    return files[0]

# heurystyki nazw kolumn
CANDIDATES = {
    "exposure":  [r"^expo", r"exposure", r"duration", r"pol.*months", r"pol.*duration"],
    "claim_cnt": [r"^n?claims?$", r"claim_count", r"freq", r"nbr_sin", r"n_sin", r"sin_num"],
    "claim_amt": [r"^claim_?amount", r"sev", r"cost", r"paid", r"sin_?cost", r"loss", r"indemn"],
    "premium":   [r"^prem", r"premium", r"prima", r"net_premium", r"earned_premium", r"tariff"],
    "age_drv":   [r"driver.*age", r"age_?driver", r"^age$", r"conducteur.*age"],
    "veh_age":   [r"veh.*age", r"age_?veh", r"car.*age"],
    "veh_power": [r"kw", r"power", r"puissance", r"cv"],
    "region":    [r"region", r"area", r"depart", r"province", r"state", r"postal", r"code_insee"],
}

def guess_columns(cols):
    lower = {c.lower(): c for c in cols}
    mapping = {}
    for key, patterns in CANDIDATES.items():
        found = None
        for p in patterns:
            for lc, orig in lower.items():
                if re.search(p, lc):
                    found = orig
                    break
            if found: break
        mapping[key] = found
    return mapping

def main():
    csv_path = pick_main_csv()
    print(f"[INFO] Using file: {csv_path.name}")

    # szybkie wczytanie (pandas sam zgadnie separator)
    df = pd.read_csv(csv_path)
    print("\n[HEAD]")
    print(df.head(10))
    print("\n[DTYPES]")
    print(df.dtypes)

    # missingi (top 20)
    print("\n[MISSING TOP 20]")
    miss = df.isna().mean().sort_values(ascending=False).head(20)
    print(miss)

    mapping = guess_columns(df.columns)
    print("\n[GUESS] suggested column mapping:")
    for k, v in mapping.items():
        print(f"  {k:10s} -> {v}")

    # podgląd podstawowych rozkładów jeśli się udało coś znaleźć
    basics = {}
    if mapping["exposure"] in df:
        basics["exposure_sum"] = float(df[mapping["exposure"]].sum())
    if mapping["claim_cnt"] in df:
        basics["claims_sum"] = float(df[mapping["claim_cnt"]].sum())
        basics["claims_rate"] = float(df[mapping["claim_cnt"]].sum() / max(1.0, df.shape[0]))
    if mapping["claim_amt"] in df:
        basics["severity_mean_on_paid"] = float(df.loc[df[mapping["claim_amt"]]>0, mapping["claim_amt"]].mean())
    if mapping["premium"] in df:
        basics["premium_mean"] = float(df[mapping["premium"]].mean())

    if basics:
        print("\n[BASICS]")
        for k, v in basics.items():
            print(f"  {k}: {v:,.4f}")

    # zapisz małą próbkę i mapping do dalszej pracy
    sample = df.sample(min(10000, len(df)), random_state=42)
    sample_path = OUT_DIR / "french_sample.parquet"
    sample.to_parquet(sample_path, index=False)
    print(f"\n[OK] sample saved -> {sample_path}")

    # mapping jako prosty csv (klucz,kolumna)
    map_path = OUT_DIR / "french_mapping.csv"
    pd.Series(mapping).to_csv(map_path, header=False)
    print(f"[OK] mapping saved -> {map_path}")

if __name__ == "__main__":
    main()
