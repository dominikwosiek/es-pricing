from pathlib import Path
from urllib.parse import urlparse
import requests
from ruamel.yaml import YAML

def main():
    # ścieżki
    config_path = Path("config/datasources.yaml")
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    # wczytaj config
    yaml = YAML(typ="safe")
    cfg = yaml.load(config_path.read_text())
    urls = cfg.get("urls", [])

    if not urls:
        print("Brak linków w config/datasources.yaml (sekcja 'urls').")
        return

    # pobierz każdy plik
    for url in urls:
        name = Path(urlparse(url).path).name or "download"
        out_path = out_dir / name
        if out_path.exists():
            print(f"[SKIP] {name} już jest")
            continue
        print(f"[GET ] {url}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        print(f"[OK  ] zapisano -> {out_path}")

    print("Gotowe.")

if __name__ == "__main__":
    main()
