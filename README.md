# es-pricing — motor TPL frequency (FR)

End-to-end pricing: preprocessing (capping), GLM (Poisson/NB) + LightGBM, analizy (Gini, Poisson dev.), Lorenz, kalibracja, raporty HTML.

## Struktura
src/es_pricing/
  data/         - fetch/preprocess
  models/       - freq models (GLM/GBM)
  analysis/     - metryki, wykresy, kalibracja
  reports/      - raporty HTML + bundle.zip
data/           - dane lokalne (ignorowane w git)

## Szybki start
(załóż, że surowe CSV są w data/raw/)
    python -m es_pricing.data.preprocess_french
    python -m es_pricing.models.freq
    python -m es_pricing.analysis.freq_analysis
    python -m es_pricing.reports.freq_report
    python -m es_pricing.reports.freq_model_report

## Wymagania
Python 3.11 + pakiety z requirements.txt
