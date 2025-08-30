
# data_module.py
import streamlit as st
import pandas as pd
import numpy as np

def load_example_data():
    """Generuje przykładowy DataFrame do demonstracji."""
    np.random.seed(42)
    N = 500
    age = np.random.randint(18, 80, N)
    gender = np.random.choice(['F', 'M'], N)
    region = np.random.choice(['North', 'South', 'East', 'West'], N)
    income = np.random.normal(50000, 15000, N).astype(int)
    exposure = np.random.rand(N)  # ekspozycja w latach (0-1)
    # Generowanie prawdopodobieństwa zdarzenia w zależności od cech (np. ryzyka ubezpieczeniowego)
    base_rate = 0.2  # bazowe prawdopodobieństwo ~20%
    logit = np.log(base_rate/(1-base_rate))
    # Wpływ cech na logit (ryzyko) - dla symulacji
    logit += 0.05 * ((age - age.mean())/10.0)        # wiek: starszy nieco zwiększa ryzyko
    logit += np.where(gender == 'M', 0.5, -0.5)      # płeć: M zwiększa, F zmniejsza logit
    region_effects = {'North': 0.3, 'South': -0.2, 'East': 0.5, 'West': -0.3}
    logit += [region_effects[r] for r in region]     # region: wschód/północ wyższe ryzyko, południe niższe
    # Uwaga: ekspozycja wpływa na prawdopodobieństwo zdarzenia - krótsza ekspozycja zmniejsza prawdopodobieństwo
    offset = np.where(exposure > 0, np.log(exposure), -10)  # log-ekspozycji (bardzo niska wartość dla ekspozycji 0)
    logit += offset
    # Obliczenie prawdopodobieństwa i wygenerowanie zmiennej celu (0/1) na podstawie rozkładu Bernoulliego
    p = 1 / (1 + np.exp(-logit))
    p = np.clip(p, 0, 1)
    target = np.random.binomial(1, p)
    # Tworzenie DataFrame
    df = pd.DataFrame({
        'target': target,
        'exposure': exposure,
        'age': age,
        'gender': gender,
        'region': region,
        'income': income
    })
    return df

def show_data_module():
    st.header("Moduł danych")
    # Jeśli nie ma jeszcze danych w sesji, tworzymy pusty DataFrame jako domyślny
    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()
    df = st.session_state.df

    # Wybór źródła danych: przykład albo plik użytkownika
    source = st.radio("Źródło danych:", ["Przykładowe dane", "Wczytaj plik (CSV/Parquet)"])
    if source == "Wczytaj plik (CSV/Parquet)":
        file = st.file_uploader("Wybierz plik z danymi", type=['csv', 'parquet'])
        if file:
            # Wczytywanie pliku do DataFrame
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_parquet(file)
            st.session_state.df = df  # zapisanie w sesji
    else:
        # Użycie danych przykładowych
        if df.empty:  # generuj tylko raz, jeśli brak w sesji
            df = load_example_data()
            st.session_state.df = df

    # Jeśli po powyższym nie mamy danych, przerywamy
    if df is None or df.empty:
        st.info("Załaduj dane, aby przejrzeć i edytować.")
        return

    st.subheader("Podgląd i edycja danych")
    # Interaktywny edytor danych - pozwala edytować wartości, dodawać wiersze
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    # Zapisz zmiany (jeśli były) z edytora do st.session_state.df
    if not edited_df.equals(df):
        st.session_state.df = edited_df
        df = edited_df

    # Wybór kolumn: target (cel), exposure (ekspozycja), random split (podział train/test)
    with st.expander("Wybór kolumn specjalnych", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            target_col = st.selectbox("Kolumna celu (target):", options=[None] + list(df.columns), index=0 if 'target' not in df.columns else (list(df.columns).index('target')+1))
        with col2:
            exposure_col = st.selectbox("Kolumna ekspozycji (opcjonalnie):", options=[None] + list(df.columns), index=0 if 'exposure' not in df.columns else (list(df.columns).index('exposure')+1))
        with col3:
            split_col = st.selectbox("Kolumna podziału train/test (opcjonalnie):", options=[None] + list(df.columns))
        # Zapisanie wyborów w stanie sesji do wykorzystania w innych modułach
        st.session_state.target_col = target_col if target_col is not None else None
        st.session_state.exposure_col = exposure_col if exposure_col is not None else None
        st.session_state.split_col = split_col if split_col is not None else None
        # Możliwość wygenerowania losowego podziału, jeśli nie ma istniejącej kolumny split
        if split_col is None:
            frac = st.slider("Udział danych treningowych (losowy podział)", min_value=0.1, max_value=0.9, value=0.75, step=0.05)
            if st.button("Wygeneruj kolumnę podziału losowego"):
                # Tworzymy kolumnę 'split' z losowym podziałem według zadanego odsetka
                df = st.session_state.df
                df['split_random'] = np.where(np.random.rand(len(df)) < frac, 'train', 'test')
                st.session_state.df = df
                st.session_state.split_col = 'split_random'
                st.success(f"Wygenerowano kolumnę 'split_random' z {int(frac*100)}% wierszy oznaczonych jako train.")
                # Odśwież interfejs (ponowne wyświetlenie selectboxów z nową kolumną)
                st.experimental_rerun()

    # Podstawowe statystyki tabelaryczne
    st.subheader("Podstawowe statystyki")
    # Tabela opisowa dla kolumn numerycznych
    desc = df.describe().T  # opis statystyczny (count, mean, std, min, quartile, max)
    # Dodajemy medianę jako oddzielny wiersz (50% to mediana już w describe)
    #desc['median'] = df.median(numeric_only=True)  # opcjonalnie, 50% z describe to mediana
    st.table(desc)

    # Wybór kolumny do szczegółowej analizy
    col_list = list(df.columns)
    selected_col = st.selectbox("Wybierz kolumnę do analizy szczegółowej:", options=col_list)
    if selected_col:
        col_data = df[selected_col]
        # Ustal typ kolumny
        numeric = False
        try:
            # Sprawdź czy da się przekonwertować na numeric (jeśli tak, potraktuj jako numericzną dla statystyk)
            pd.to_numeric(col_data.dropna().iloc[:5])
            numeric = True
        except Exception:
            numeric = False

        with st.expander(f"Statystyki kolumny: {selected_col}", expanded=True):
            # Wyświetl podstawowe statystyki dla tej kolumny
            if numeric:
                st.write(f"**Min:** {col_data.min()}  \n**Max:** {col_data.max()}  \n**Średnia:** {col_data.mean():.3f}  \n**Mediana:** {col_data.median():.3f}")
                # Percentyle
                percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
                perc_values = col_data.quantile(percentiles).to_dict()
                perc_str = "  \n".join([f"**{int(p*100)}%:** {v:.3f}" for p, v in perc_values.items()])
                st.write("**Percentyle:**  \n" + perc_str)
            else:
                # Dla kolumn nienumerycznych: liczba unikalnych kategorii, najczęstsza itp.
                unique_vals = col_data.unique()
                st.write(f"Liczba unikalnych wartości: {len(unique_vals)}")
                # Najczęstsze wartości
                top_counts = col_data.value_counts().head(5)
                st.write("Top 5 wartości:")
                st.table(pd.DataFrame({selected_col: top_counts.index, "Count": top_counts.values}))

            # Histogram/rozklad wartości kolumny
            st.write("**Rozkład wartości:**")
            if numeric:
                # Opcja ważenia ekspozycją (jeśli dotyczy) dla histogramu
                weight_option = None
                if st.session_state.exposure_col and st.session_state.exposure_col in df.columns:
                    weight_choice = st.radio("Ważenie histogramu:", ["jednostkowy", "ekspozycją"], index=0, horizontal=True)
                    if weight_choice == "ekspozycją":
                        weight_option = df[st.session_state.exposure_col]
                # Wykorzystujemy numpy do histogramu
                values = col_data.dropna().values
                weights = None
                if weight_option is not None:
                    weights = weight_option.loc[col_data.index].dropna().values
                    # Dopasuj długość wektorów po odrzuceniu NA
                    if len(weights) != len(values):
                        weights = weights[:len(values)]
                hist, bins = np.histogram(values, bins=20, weights=weights)
                # Normalizacja do 1 (rozkład względny) jeśli jednostkowy, lub do sumy ekspozycji (jeśli ważony)
                if weight_option is None:
                    hist_percent = hist / hist.sum()
                    ylabel = "Udział liczby obserwacji"
                else:
                    hist_percent = hist / (weights.sum() if weights is not None else 1)
                    ylabel = "Udział ekspozycji"
                # Środek binów do wykresu
                bin_centers = (bins[:-1] + bins[1:]) / 2
                # Tworzenie wykresu za pomocą Plotly (liniowy wykres słupkowy)
                import plotly.graph_objects as go
                fig = go.Figure(data=go.Bar(x=bin_centers, y=hist_percent, width=bins[1]-bins[0]))
                fig.update_layout(xaxis_title=selected_col, yaxis_title=ylabel, bargap=0)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Wykres słupkowy dla kategorycznych: liczność lub suma ekspozycji
                cat_counts = col_data.value_counts()
                if st.session_state.exposure_col and st.session_state.exposure_col in df.columns:
                    # Suma ekspozycji na kategorię
                    exp_sum = df.groupby(selected_col)[st.session_state.exposure_col].sum()
                    # Tworzymy ramkę z dwoma miarami
                    cat_stats = pd.DataFrame({
                        'Count': cat_counts,
                        'Total_exposure': exp_sum
                    })
                    st.write("Liczność oraz suma ekspozycji dla każdej kategorii:")
                    st.table(cat_stats)
                    # Wykres sumy ekspozycji per kategoria
                    st.write("**Wykres: suma ekspozycji w poszczególnych kategoriach**")
                    import plotly.express as px
                    fig = px.bar(x=exp_sum.index, y=exp_sum.values, labels={'x': selected_col, 'y': 'Suma ekspozycji'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Liczność kategorii:")
                    st.table(pd.DataFrame({selected_col: cat_counts.index, "Count": cat_counts.values}))
                    # Wykres liczności
                    import plotly.express as px
                    fig = px.bar(x=cat_counts.index, y=cat_counts.values, labels={'x': selected_col, 'y': 'Liczność'})
                    st.plotly_chart(fig, use_container_width=True)

    # Zaawansowane operacje na danych: zmiana typów, dodawanie kolumn
    with st.expander("Zaawansowane operacje na danych"):
        st.write("**Zmiana typu danych kolumny:**")
        col_to_change = st.selectbox("Wybierz kolumnę do zmiany typu:", options=[None] + list(df.columns))
        new_type = st.selectbox("Nowy typ danych:", options=["int", "float", "str", "category"])
        if st.button("Zmień typ"):
            if col_to_change:
                try:
                    if new_type == "int":
                        df[col_to_change] = pd.to_numeric(df[col_to_change], errors='coerce').astype("Int64")
                    elif new_type == "float":
                        df[col_to_change] = pd.to_numeric(df[col_to_change], errors='coerce').astype("Float64")
                    elif new_type == "str":
                        df[col_to_change] = df[col_to_change].astype(str)
                    elif new_type == "category":
                        df[col_to_change] = df[col_to_change].astype('category')
                    st.session_state.df = df
                    st.success(f"Zmieniono typ kolumny **{col_to_change}** na {new_type}.")
                except Exception as e:
                    st.error(f"Nie udało się zmienić typu: {e}")
        st.write("---")
        st.write("**Dodawanie nowej kolumny:**")
        new_col_name = st.text_input("Nazwa nowej kolumny:")
        new_col_value = st.text_input("Wartość początkowa (jednakowa dla wszystkich wierszy):", value="")
        if st.button("Dodaj kolumnę"):
            if new_col_name:
                val = None
                if new_col_value == "":
                    val = np.nan
                else:
                    # Spróbuj odgadnąć typ wartości (int, float lub pozostaw jako string)
                    try:
                        # jeśli wartość zawiera np. kropkę lub wygląda jak liczba - rzutuj
                        if new_col_value.isdigit():
                            val = int(new_col_value)
                        else:
                            val = float(new_col_value)
                    except:
                        val = new_col_value  # traktuj jako tekst
                df[new_col_name] = val
                st.session_state.df = df
                st.success(f"Dodano nową kolumnę **{new_col_name}**.")
