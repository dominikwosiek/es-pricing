from __future__ import annotations
from pathlib import Path
import subprocess
import sys
import json
import shutil

def _kaggle_json_ok() -> bool:
    # Windows: C:\Users\<user>\.kaggle\kaggle.json
    kaggle_dir = Path.home() / ".kaggle"
    f = kaggle_dir / "kaggle.json"
    if not f.exists():
        print(f"[ERR ] Brak pliku {f}. Wejdź na https://www.kaggle.com/account → Create New API Token.")
        return False
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        if not data.get("username") or not data.get("key"):
            print("[ERR ] kaggle.json istnieje, ale nie ma 'username'/'key'.")
            return False
    except Exception as e:
        print(f"[ERR ] Nie mogę odczytać kaggle.json: {e}")
        return False
    # uprawnienia (na Windows mniej krytyczne, ale sprawdźmy katalog)
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    return True

def _kaggle_cli_ok() -> bool:
    if shutil.which("kaggle") is None:
        print("[ERR ] Komenda 'kaggle' nieznaleziona. Zainstaluj: pip install kaggle (i zrestartuj terminal/PyCharm).")
        return False
    return True

def main():
    if not _kaggle_cli_ok() or not _kaggle_json_ok():
        sys.exit(1)

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = "karansarpal/fremtpl2-french-motor-tpl-insurance-claims"
    print(f"[GET ] Kaggle dataset: {ds}")
    # pobierz i rozpakuj wprost do data/raw
    cmd = [
        "kaggle", "datasets", "download",
        "-d", ds,
        "-p", str(out_dir),
        "--unzip"
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERR ] Pobieranie nie powiodło się (kod {e.returncode}).")
        sys.exit(e.returncode)

    # Podpowiedź nazw plików, które zwykle lądują
    found = list(out_dir.glob("*.csv")) + list(out_dir.glob("*.parquet")) + list(out_dir.glob("*.xlsx"))
    if found:
        print("[OK  ] Zapisano pliki:")
        for p in found:
            print("       -", p.name)
    else:
        print("[WARN] Nie widzę plików .csv/.parquet/.xlsx w data/raw. Sprawdź zawartość archiwum ręcznie.")

if __name__ == "__main__":
    main()
