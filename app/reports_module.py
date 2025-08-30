
# reports_module.py
import streamlit as st
import pandas as pd
import base64

def show_reports_module():
    st.header("Moduł raportów")
    # Sprawdź czy istnieją dane i wyniki modeli
    if 'df' not in st.session_state or st.session_state.df is None or st.session_state.df.empty:
        st.info("Brak danych do wygenerowania raportu. Wczytaj dane w module Data.")
        return

    df = st.session_state.df
    target_col = st.session_state.get('target_col', None)
    exposure_col = st.session_state.get('exposure_col', None)
    split_col = st.session_state.get('split_col', None)
    glm_results = st.session_state.get('glm_results', None)
    gbm_model = st.session_state.get('gbm_model', None)
    glm_metrics = st.session_state.get('glm_metrics', {})
    gbm_metrics = st.session_state.get('gbm_metrics', {})

    # Sekcja danych - informacje ogólne
    report_html = "<h1>Raport z analiz i modeli</h1>"
    report_html += "<h2>1. Dane</h2>"
    report_html += f"<p>Liczba obserwacji: <b>{len(df)}</b><br>"
    report_html += f"Liczba kolumn: <b>{df.shape[1]}</b><br>"
    report_html += f"Kolumna celu (target): <b>{target_col or 'nie wybrano'}</b><br>"
    report_html += f"Kolumna ekspozycji: <b>{exposure_col or '(brak)'}</b><br>"
    report_html += f"Kolumna podziału train/test: <b>{split_col or '(brak)'}</b></p>"

    # Sekcja modelu GLM
    report_html += "<h2>2. Model GLM</h2>"
    if glm_results:
        # Dodaj współczynniki i p-value do raportu (tabela HTML)
        params = glm_results.params
        pvalues = glm_results.pvalues
        coef_df = pd.DataFrame({"coef": params, "p-value": pvalues})
        report_html += "<h3>Współczynniki modelu GLM</h3>"
        report_html += coef_df.to_html(float_format="%.3f")
        # Dodaj metryki modelu
        if glm_metrics:
            report_html += "<p><b>Metryki (GLM, zbiór testowy):</b><br>"
            if "AUC" in glm_metrics:
                report_html += f"AUC = {glm_metrics['AUC']:.3f}, Gini = {glm_metrics.get('Gini', float('nan')):.3f}<br>"
            if "Deviance" in glm_metrics:
                report_html += f"Dewiancja = {glm_metrics['Deviance']:.2f}<br>"
            if "RMSE" in glm_metrics:
                report_html += f"RMSE = {glm_metrics['RMSE']:.3f}<br>"
            report_html += "</p>"
    else:
        report_html += "<p><i>Model GLM nie został zbudowany.</i></p>"

    # Sekcja modelu GBM
    report_html += "<h2>3. Model GBM</h2>"
    if gbm_model:
        # Ważność cech modelu GBM
        importances = gbm_model.feature_importances_
        feature_names = gbm_model.feature_name_ if hasattr(gbm_model, 'feature_name_') else None
        if feature_names is None or len(feature_names) != len(importances):
            feature_names = st.session_state.get('selected_features', list(range(len(importances))))
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)
        report_html += "<h3>Ważność cech (GBM)</h3>"
        report_html += imp_df.to_html(index=False, float_format="%.2f")
        # Metryki modelu GBM
        if gbm_metrics:
            report_html += "<p><b>Metryki (GBM, zbiór testowy):</b><br>"
            if "AUC" in gbm_metrics:
                report_html += f"AUC = {gbm_metrics['AUC']:.3f}, Gini = {gbm_metrics.get('Gini', float('nan')):.3f}<br>"
            if "LogLoss" in gbm_metrics:
                report_html += f"Log-loss = {gbm_metrics['LogLoss']:.3f}<br>"
            if "RMSE" in gbm_metrics:
                report_html += f"RMSE = {gbm_metrics['RMSE']:.3f}<br>"
            report_html += "</p>"
    else:
        report_html += "<p><i>Model GBM nie został zbudowany.</i></p>"

    # Sekcja porównania modeli
    report_html += "<h2>4. Porównanie modeli</h2>"
    if glm_results and gbm_model and glm_metrics and gbm_metrics:
        report_html += "<p><b>Porównanie wybranych metryk (zbiór testowy):</b><br>"
        if "Gini" in glm_metrics and "Gini" in gbm_metrics:
            report_html += f"Gini GLM = {glm_metrics['Gini']:.3f}, Gini GBM = {gbm_metrics['Gini']:.3f}<br>"
        if "AUC" in glm_metrics and "AUC" in gbm_metrics:
            report_html += f"AUC GLM = {glm_metrics['AUC']:.3f}, AUC GBM = {gbm_metrics['AUC']:.3f}<br>"
        if "Deviance" in glm_metrics and "LogLoss" in gbm_metrics:
            report_html += f"Dewiancja GLM = {glm_metrics['Deviance']:.2f}, Log-loss GBM = {gbm_metrics['LogLoss']:.2f}<br>"
        if "RMSE" in glm_metrics and "RMSE" in gbm_metrics:
            report_html += f"RMSE GLM = {glm_metrics['RMSE']:.3f}, RMSE GBM = {gbm_metrics['RMSE']:.3f}"
        report_html += "</p>"
    else:
        report_html += "<p><i>Brak wyników obu modeli do pełnego porównania.</i></p>"

    # Sekcja wykresów: kalibracja i Lorenz (jeśli dostępne w sesji)
    if 'fig_calibration' in st.session_state or 'fig_lorenz' in st.session_state:
        report_html += "<h2>5. Wykresy</h2>"
    if 'fig_calibration' in st.session_state:
        fig_cal = st.session_state.fig_calibration
        # Konwersja wykresu plotly na obraz PNG (base64)
        try:
            img_bytes = fig_cal.to_image(format="png", engine="kaleido")
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            report_html += '<h3>Wykres kalibracji</h3>'
            report_html += f'<img src="data:image/png;base64,{b64}" style="max-width:700px;"><br>'
        except Exception as e:
            report_html += "<p>(Wykres kalibracji niedostępny - brak wsparcia dla eksportu obrazu)</p>"
    if 'fig_lorenz' in st.session_state:
        fig_lorenz = st.session_state.fig_lorenz
        try:
            img_bytes = fig_lorenz.to_image(format="png", engine="kaleido")
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            report_html += '<h3>Krzywe Lorenz</h3>'
            report_html += f'<img src="data:image/png;base64,{b64}" style="max-width:700px;"><br>'
        except Exception as e:
            report_html += "<p>(Wykres Lorenz niedostępny - brak wsparcia dla eksportu obrazu)</p>"

    # Wyświetlenie podglądu raportu HTML
    st.subheader("Podgląd raportu")
    st.components.v1.html(report_html, height=400, scrolling=True)
    # Przycisk pobrania raportu HTML
    st.download_button("📥 Pobierz raport HTML", data=report_html.encode('utf-8'), file_name="raport.html", mime="text/html")
    # Opcja generowania PDF (wymaga kaleido zainstalowanego)
    try:
        # Próba importu kaleido (silnik używany przez plotly do renderowania obrazów)
        import kaleido  # jeśli nie zainstalowano, rzuci wyjątek
        import pdfkit   # opcjonalnie, jeśli zainstalowany wkhtmltopdf
        # Konwersja HTML na PDF (jeśli pdfkit dostępny)
        pdf_bytes = None
        try:
            pdf_bytes = pdfkit.from_string(report_html, False)
        except Exception:
            pdf_bytes = None
        if pdf_bytes:
            st.download_button("📥 Pobierz PDF", data=pdf_bytes, file_name="raport.pdf", mime="application/pdf")
        else:
            st.info("Aby wygenerować PDF, zainstaluj w systemie wkhtmltopdf i bibliotekę pdfkit.")
    except ImportError:
        st.caption("Info: Możesz zainstalować bibliotekę 'kaleido' aby umożliwić zapis wykresów do PDF.")
