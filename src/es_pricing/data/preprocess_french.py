from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    freq_path = RAW / "freMTPL2freq.csv"
    sev_path  = RAW / "freMTPL2sev.csv"

    if not freq_path.exists() or not sev_path.exists():
        raise SystemExit("Brakuje plików freMTPL2freq.csv / freMTPL2sev.csv w data/raw/")

    # === wczytaj surowe ===
    freq = pd.read_csv(freq_path)
    sev  = pd.read_csv(sev_path)

    print("[INFO] freq shape:", freq.shape)
    print("[INFO] sev  shape:", sev.shape)

    # === severity na poziom polisy ===
    sev_sum = (
        sev.groupby("IDpol", as_index=False)["ClaimAmount"]
           .sum()
           .rename(columns={"ClaimAmount": "claim_amount"})
    )

    # === JOIN polis + suma szkód ===
    merged = freq.merge(sev_sum, on="IDpol", how="left")
    merged["claim_amount"] = merged["claim_amount"].fillna(0.0)

    # === angielskie nazwy + wybór kolumn ===
    df_raw = merged.rename(columns={
        "IDpol": "policy_id",
        "ClaimNb": "claim_count",
        "Exposure": "exposure",
        "DrivAge": "driver_age",
        "VehAge": "vehicle_age",
        "VehPower": "vehicle_power",
        "Region": "region",
        "Area": "area",
        "Density": "density",
        "VehBrand": "vehicle_brand",
        "VehGas": "fuel",
        "BonusMalus": "bonus_malus",
    })[
        ["policy_id","claim_count","exposure","claim_amount",
         "driver_age","vehicle_age","vehicle_power",
         "region","area","density","vehicle_brand","fuel","bonus_malus"]
    ].copy()

    # typy liczbowe
    df_raw["claim_count"] = pd.to_numeric(df_raw["claim_count"], errors="coerce").fillna(0).astype(int)
    df_raw["exposure"] = pd.to_numeric(df_raw["exposure"], errors="coerce").fillna(0.0).astype(float)
    df_raw["claim_amount"] = pd.to_numeric(df_raw["claim_amount"], errors="coerce").fillna(0.0).astype(float)
    for c in ["driver_age","vehicle_age","vehicle_power","bonus_malus","density"]:
        if c in df_raw.columns:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

    # === quick checks (raw) ===
    print("\n[CHECKS RAW]")
    print("exposure sum:", float(df_raw["exposure"].sum()))
    print("total claims:", int(df_raw["claim_count"].sum()))
    paid_raw = df_raw.loc[df_raw["claim_amount"]>0, "claim_amount"]
    print("avg severity (on paid):", float(paid_raw.mean()) if len(paid_raw) else 0.0)

    # === capped/stabilized view (kanon dla freMTPL2) ===
    df = df_raw.copy()
    df["claim_count"] = df["claim_count"].clip(0, 4)
    df["exposure"] = df["exposure"].clip(1e-6, 1.0)
    df = df[df["exposure"] >= 0.02].copy()    # odfiltruj bardzo małą ekspozycję (stabilność GLM/GBM)

    print("\n[CHECKS CAPPED]")
    print("exposure sum:", float(df["exposure"].sum()))
    print("total claims:", int(df["claim_count"].sum()))
    paid = df.loc[df["claim_amount"]>0, "claim_amount"]
    print("avg severity (on paid):", float(paid.mean()) if len(paid) else 0.0)

    # === zapisy ===
    out_raw = OUT / "french.parquet"
    df_raw.to_parquet(out_raw, index=False)
    print(f"[OK] raw saved -> {out_raw}")

    out_cap = OUT / "french_capped.parquet"
    df.to_parquet(out_cap, index=False)
    print(f"[OK] capped saved -> {out_cap}")
    print(df.head())

if __name__ == "__main__":
    main()
