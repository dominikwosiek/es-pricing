
# risk_module.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def show_risk_module():
    st.header("Moduł eksploracji ryzyka")
    # Sprawdzenie czy dane i target są dostępne
    if 'df' not in st.session_state or st.session_state.df is None or st.session_state.df.empty:
        st.warning("Brak danych - proszę załadować dane w zakładce Data.")
        return
    if 'target_col' not in st.session_state or st.session_state.target_col is None:
        st.info("Najpierw wybierz kolumnę celu (target) w sekcji Data.")
        return

    df = st.session_state.df
    target_col = st.session_state.target_col
    exposure_col = st.session_state.exposure_col

    # Wybór zmiennej niezależnej do analizy
    variables = [col for col in df.columns if col != target_col]
    selected_var = st.selectbox("Wybierz zmienną do analizy ryzyka:", options=variables)
    if not selected_var:
        return

    # Przygotowanie danych do wykresu zależności target vs selected_var
    var_data = df[selected_var]
    # Sprawdzenie typu zmiennej (numeryczna czy kategoria)
    if pd.api.types.is_numeric_dtype(var_data) and var_data.nunique() > 10:
        # Jeśli numeryczna z wieloma wartościami - binning
        bins = st.slider("Liczba przedziałów (binów):", min_value=2, max_value=20, value=5)
        # Używamy qcut (podział na kwantyle) aby grupy miały podobną liczność
        try:
            var_bins = pd.qcut(var_data, q=bins, duplicates='drop')
        except Exception:
            # qcut może czasem nie rozróżnić binów, wówczas używamy cut z równymi odstępami
            var_bins = pd.cut(var_data, bins)
        grouped = df.groupby(var_bins)
        x_labels = [str(interval) for interval in grouped.indices.keys()]
    else:
        # Dla zmiennych kategorycznych lub numerycznych z niewieloma unikalnymi wartościami
        grouped = df.groupby(selected_var)
        x_labels = list(grouped.indices.keys())

    # Oblicz średni target w każdej grupie (ważony ekspozycją lub nie)
    avg_target = []
    for name, group in grouped:
        if exposure_col and exposure_col in group.columns and group[exposure_col].sum() > 0:
            # Średnia ważona ekspozycją: suma target / suma ekspozycji
            val = (group[target_col] * group[exposure_col]).sum() / group[exposure_col].sum()
        else:
            val = group[target_col].mean()
        avg_target.append(val)
    avg_target = np.array(avg_target)

    # Tworzenie wykresu zależności
    fig = px.line(x=x_labels, y=avg_target, markers=True) if len(x_labels) > 1 else px.bar(x=x_labels, y=avg_target)
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(xaxis_title=selected_var, yaxis_title=f"Średnia {target_col}", title=f"Średnia {target_col} względem {selected_var}")
    st.plotly_chart(fig, use_container_width=True)

    # Korelacja między zmienną a celem (jeśli numeryczna)
    if pd.api.types.is_numeric_dtype(var_data) or pd.api.types.is_numeric_dtype(df[target_col]):
        try:
            corr_val = df[[selected_var, target_col]].corr(numeric_only=True).iloc[0, 1]
        except Exception:
            corr_val = None
        if corr_val is not None and not np.isnan(corr_val):
            st.write(f"**Korelacja Pearsona {selected_var} vs {target_col}:** {corr_val:.3f}")

    # Dodatkowo: opcjonalne pokazanie korelacji pomiędzy wszystkimi zmiennymi numerycznymi
    with st.expander("Macierz korelacji zmiennych numerycznych"):
        num_df = df.select_dtypes(include=[int, float])
        if num_df.shape[1] < 2:
            st.write("Brak wystarczającej liczby zmiennych numerycznych do obliczenia korelacji.")
        else:
            corr_matrix = num_df.corr(numeric_only=True)
            st.write("Macierz korelacji (współczynniki Pearsona):")
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"), height=300)
