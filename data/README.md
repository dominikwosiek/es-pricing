# Data folder

Pliki danych nie są wersjonowane (patrz .gitignore). Zostawiamy tu tylko ten README.

Struktura:
- data/raw/       - wgraj surowe CSV: freMTPL2freq.csv, freMTPL2sev.csv
- data/processed/ - powstaje po preprocessie (*.parquet)
- data/interim/   - predykcje, metryki, wykresy, raporty HTML, modele (*.pkl)

Odtworzenie pipeline'u (lokalnie):
    python -m es_pricing.data.preprocess_french
    python -m es_pricing.models.freq
    python -m es_pricing.analysis.freq_analysis
    python -m es_pricing.reports.freq_report
    python -m es_pricing.reports.freq_model_report
 
