#  Frequency pricing  + Streamlit app  
_Utworzone z pomocą ChatGPT_

Interaktywny projekt do modelowania częstości szkód (frequency) dla ubezpieczeń komunikacyjnych na danych **freMTPL2** (Francja).  
Zawiera pipeline danych, GLM/GBM, analizy (Gini, dewiancja, kalibracja, Lorenz), raporty oraz **aplikację Streamlit** do przeglądania/edycji danych i porównywania modeli.

---

## Funkcje
- Pobranie i przygotowanie danych (Kaggle freMTPL2)
- Modelowanie **Poisson GLM / NegBin GLM / GBM (LightGBM)**
- Metryki: Poisson dev., Gini, AUC 
- Raporty HTML (opcjonalnie PDF)
- **Aplikacja Streamlit**: Data / Risk / Model / Reports
- Edycja danych w UI (typy, wartości, nowe kolumny), statystyki, histogramy
 
---

## Wymagania
- Python 3.11  
- Windows/Linux/macOS  
- (opcjonalnie) konto Kaggle + `kaggle.json` w `%USERPROFILE%\.kaggle\`  

Pakiety (skrótem): `pandas`, `numpy`, `statsmodels`, `scikit-learn`, `lightgbm`, `plotly`, `streamlit`, `pyarrow`, `kaleido` (do PNG/raportów).

---

## Szybki start (Windows / PowerShell)
```powershell
cd "C:\Users\Dominik\OneDrive\Pulpit\es-pricing"
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit plotly lightgbm kaleido
```

Pobranie danych (Kaggle):

Umieść kaggle.json w: C:\Users\<USER>\.kaggle\kaggle.json

Uruchom:

```powershell
Skopiuj kod
python -m es_pricing.data.fetch_french
python -m es_pricing.data.preprocess_french
```

Trening modeli + analizy + raporty (CLI):

```powershell
Skopiuj kod
python -m es_pricing.models.freq
python -m es_pricing.analysis.freq_analysis
python -m es_pricing.reports.freq_report
python -m es_pricing.reports.freq_model_report
```
Aplikacja Streamlit:

```powershell
Skopiuj kod
python -m streamlit run app/app.py
```
Przeglądarka: http://localhost:8501

## Notatki o danych
Źródło: Kaggle – freMTPL2 (FR MTPL, częstotliwość i severities)

Pliki raw trafiają do data/raw/ (nie są trzymane w repo)
  