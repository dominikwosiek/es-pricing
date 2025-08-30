
# model_module.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import statsmodels.api as sm
from lightgbm import LGBMClassifier, LGBMRegressor

def show_model_module():
    st.header("Moduł modelowania")
    # Walidacja dostępności danych i wyboru kolumn
    if 'df' not in st.session_state or st.session_state.df is None or st.session_state.df.empty:
        st.warning("Brak danych - najpierw wczytaj dane w zakładce Data.")
        return
    if 'target_col' not in st.session_state or st.session_state.target_col is None:
        st.info("Wybierz kolumnę celu (target) w zakładce Data, aby trenować modele.")
        return

    df = st.session_state.df
    target_col = st.session_state.target_col
    exposure_col = st.session_state.exposure_col
    split_col = st.session_state.split_col

    # Podział na train/test
    if split_col and split_col in df.columns:
        # Jeżeli kolumna split jest liczbowa i zawiera wiele wartości (np. losowa 0-1) => stosujemy próg 0.5 lub 0.8
        if df[split_col].dtype in [float, int] and df[split_col].nunique() > 2:
            threshold = 0.8
            train_idx = df[split_col] <= df[split_col].quantile(threshold)
            test_idx = ~train_idx
        else:
            # Zakładamy 2 unikalne wartości oznaczające train vs test
            unique_vals = df[split_col].unique().tolist()
            if len(unique_vals) == 2:
                # Pozwól użytkownikowi wybrać która wartość oznacza zbiór treningowy
                train_label = st.selectbox("Wartość w kolumnie split oznaczająca zbiór treningowy:", options=unique_vals, index=0)
                test_label = [val for val in unique_vals if val != train_label][0]
                train_idx = df[split_col] == train_label
                test_idx = df[split_col] == test_label
            else:
                # Jeśli więcej niż 2 wartości unikalne, traktujemy pierwszą jako train, resztę jako test (upraszczenie)
                train_label = unique_vals[0]
                train_idx = df[split_col] == train_label
                test_idx = ~train_idx
        train_df = df[train_idx].copy()
        test_df = df[test_idx].copy()
        st.write(f"Podział danych: {len(train_df)} obserwacji treningowych, {len(test_df)} testowych (wg kolumny **{split_col}**).")
    else:
        # Brak kolumny podziału - dzielimy losowo 75/25
        train_df = df.sample(frac=0.75, random_state=42)
        test_df = df.drop(train_df.index)
        st.write(f"Użyto losowego podziału: {len(train_df)} obserwacji treningowych, {len(test_df)} testowych.")

    # Wybór listy cech do modelu (poza target i ewentualnie exposure)
    features = [col for col in df.columns if col != target_col and col != exposure_col and col != split_col]
    selected_features = st.multiselect("Wybierz zmienne do modeli:", options=features, default=features)
    if not selected_features:
        st.error("Nie wybrano zmiennych objaśniających.")
        return

    # Przygotowanie formuły dla GLM (uwzględniamy kategoryczne przez C())
    formula_terms = []
    for col in selected_features:
        if pd.api.types.is_numeric_dtype(df[col]):
            formula_terms.append(f"{col}")
        else:
            formula_terms.append(f"C({col})")  # oznaczenie zmiennej jako kategoria w formule
    formula = f"{target_col} ~ " + " + ".join(formula_terms)

    # Sekcja modelu GLM
    st.subheader("Model GLM")
    glm_family = None
    # Domyślnie wybieramy rodzaj GLM w zależności od typu targetu
    if pd.api.types.is_numeric_dtype(df[target_col]):
        if df[target_col].nunique() == 2:
            glm_family = sm.families.Binomial()  # regresja logistyczna dla binarnego targetu
        else:
            # Dla ciągłego: użyj Gaussian (OLS) jako domyślnie
            glm_family = sm.families.Gaussian()
    else:
        # Jeśli target jest kategorią 0/1 zapisany jako str czy bool, też Binomial
        glm_family = sm.families.Binomial()

    # Parametry GLM (można rozszerzyć o wybór link function, rodzin itp.)
    if isinstance(glm_family, sm.families.Binomial):
        st.write("Rodzaj GLM: **Regresja logistyczna (Binomial)**")
    elif isinstance(glm_family, sm.families.Gaussian):
        st.write("Rodzaj GLM: **Regresja liniowa (Gaussian)**")
    else:
        st.write(f"Rodzaj GLM: {glm_family.__class__.__name__}")

    # Trening modelu GLM po kliknięciu przycisku
    glm_button = st.button("Trenuj model GLM")
    if glm_button:
        try:
            # Dopasowanie modelu GLM za pomocą statsmodels
            if exposure_col:
                # Użycie ekspozycji jako wagi (freq_weights) jeśli jest podana
                train_df_local = train_df.copy()
                # Jeżeli model Poisson byśmy robili, można by użyć exposure param z log link,
                # ale dla Binomial używamy freq_weights (aproksymacja)
                glm_results = smf.glm(formula=formula, data=train_df_local, family=glm_family,
                                      freq_weights=train_df_local[exposure_col] if exposure_col in train_df_local.columns else None).fit()
            else:
                glm_results = smf.glm(formula=formula, data=train_df, family=glm_family).fit()
            st.session_state.glm_results = glm_results  # zachowaj model
            # Wyciągnięcie współczynników i p-wartości
            params = glm_results.params
            pvalues = glm_results.pvalues
            coef_table = pd.DataFrame({"coef": params, "p-value": pvalues})
            st.write("**Współczynniki modelu (GLM) i istotność:**")
            st.table(coef_table.style.format({"coef": "{:.3f}", "p-value": "{:.3e}"}))
            # Metryki na zbiorze testowym
            # Jeśli model Binomial (klasyfikacja) - policz prawdopodobieństwa
            if isinstance(glm_family, sm.families.Binomial):
                pred_probs = glm_results.predict(test_df)
                # Czasem predykcja może dać trochę poza [0,1] ze względu na wagowanie - obetnijmy:
                pred_probs = np.clip(pred_probs, 0, 1)
                y_true = test_df[target_col].values
                auc = roc_auc_score(y_true, pred_probs)
                gini = 2*auc - 1
                deviance = 2 * log_loss(y_true, pred_probs) * len(y_true)  # dewiancja = -2*log-lik (tu log_loss * N * 2)
                st.session_state.glm_metrics = {"AUC": auc, "Gini": gini, "Deviance": deviance}
                st.write(f"**AUC (test):** {auc:.3f},  **Gini:** {gini:.3f},  **Dewiancja (test):** {deviance:.2f}")
                st.session_state.glm_pred = pred_probs
            else:
                # Jeśli model regresyjny ciągły: przewidywania i MSE/RMSE
                preds = glm_results.predict(test_df)
                y_true = test_df[target_col].values
                mse = np.mean((y_true - preds)**2)
                st.session_state.glm_metrics = {"MSE": mse, "RMSE": np.sqrt(mse)}
                st.write(f"**RMSE (test):** {np.sqrt(mse):.3f}")
                st.session_state.glm_pred = preds
            # Zachowanie y_true dla wykorzystania w porównaniach
            st.session_state.y_test = test_df[target_col].values
            # Pełne podsumowanie modelu w ekspanderze
            with st.expander("Podsumowanie modelu GLM"):
                st.text(glm_results.summary())
        except Exception as e:
            st.error(f"Błąd podczas trenowania GLM: {e}")

    # Sekcja modelu GBM
    st.subheader("Model GBM (LightGBM)")
    # Parametry modelu - interaktywne widgety dla wybranych hiperparametrów
    col1, col2, col3 = st.columns(3)
    with col1:
        num_trees = st.number_input("Liczba drzew (n_estimators)", min_value=10, max_value=500, value=100, step=10)
    with col2:
        max_depth = st.number_input("Maksymalna głębokość drzewa", min_value=-1, max_value=20, value=-1, help="Ustaw -1 dla braku ograniczenia")
    with col3:
        learning_rate = st.number_input("Learning rate", min_value=0.001, max_value=1.0, value=0.1, step=0.05)
    # Przycisk treningu GBM
    gbm_button = st.button("Trenuj model GBM")
    if gbm_button:
        try:
            # Wybór modelu LightGBM odpowiednio do typu problemu
            if df[target_col].nunique() == 2 or df[target_col].dtype == bool or df[target_col].dtype == object:
                # Problem binarnej klasyfikacji
                gbm_model = LGBMClassifier(n_estimators=int(num_trees), max_depth=int(max_depth), learning_rate=float(learning_rate), random_state=42)
                X_train = train_df[selected_features]
                y_train = train_df[target_col]
                # Dopasowanie modelu z próbkami ważonymi ekspozycją (jeśli podano)
                if exposure_col and exposure_col in train_df.columns:
                    gbm_model.fit(X_train, y_train, sample_weight=train_df[exposure_col])
                else:
                    gbm_model.fit(X_train, y_train)
                # Predykcje na zbiorze testowym (prawdopodobieństwa dla klasy pozytywnej)
                X_test = test_df[selected_features]
                y_test = test_df[target_col]
                pred_probs = gbm_model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, pred_probs)
                gini = 2*auc - 1
                loss = log_loss(y_test, pred_probs)
                st.session_state.gbm_metrics = {"AUC": auc, "Gini": gini, "LogLoss": loss}
                st.write(f"**AUC (test):** {auc:.3f},  **Gini:** {gini:.3f},  **Log-loss:** {loss:.3f}")
                st.session_state.gbm_pred = pred_probs
                st.session_state.y_test = y_test.values  # zapisujemy prawdziwe etykiety (może nadpisywać wcześniejsze, ale powinny być takie same)
            else:
                # Problem regresji (target ciągły nie-binarny)
                gbm_model = LGBMRegressor(n_estimators=int(num_trees), max_depth=int(max_depth), learning_rate=float(learning_rate), random_state=42)
                X_train = train_df[selected_features]
                y_train = train_df[target_col]
                if exposure_col and exposure_col in train_df.columns:
                    gbm_model.fit(X_train, y_train, sample_weight=train_df[exposure_col])
                else:
                    gbm_model.fit(X_train, y_train)
                X_test = test_df[selected_features]
                y_test = test_df[target_col]
                preds = gbm_model.predict(X_test)
                mse = np.mean((y_test - preds)**2)
                st.session_state.gbm_metrics = {"MSE": mse, "RMSE": np.sqrt(mse)}
                st.write(f"**RMSE (test):** {np.sqrt(mse):.3f}")
                st.session_state.gbm_pred = preds
                st.session_state.y_test = y_test.values
            st.session_state.gbm_model = gbm_model  # zapisujemy model, by móc np. użyć później
            # Istotność cech (feature importance)
            feature_names = selected_features
            importances = gbm_model.feature_importances_
            imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            st.write("**Feature importance (ważność cech w modelu GBM):**")
            st.table(pd.DataFrame({"Feature": imp_series.index, "Importance": imp_series.values}))
            # Wykres słupkowy istotności cech
            fig_imp = px.bar(x=imp_series.values[::-1], y=imp_series.index[::-1], orientation='h',
                             labels={"x": "Importance", "y": "Feature"}, title="Feature Importance - GBM")
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.error(f"Błąd podczas trenowania GBM: {e}")

    # Porównanie modeli (jeśli oba zostały wytrenowane)
    st.subheader("Porównanie modeli")
    if 'glm_metrics' in st.session_state and 'gbm_metrics' in st.session_state:
        glm_m = st.session_state.glm_metrics
        gbm_m = st.session_state.gbm_metrics
        # Porównanie na podstawie Gini (jeśli dostępny) lub RMSE/MSE dla regresji
        col1, col2 = st.columns(2)
        if "Gini" in glm_m and "Gini" in gbm_m:
            col1.metric("Gini - GLM", f"{glm_m['Gini']:.3f}")
            col2.metric("Gini - GBM", f"{gbm_m['Gini']:.3f}")
        if "AUC" in glm_m and "AUC" in gbm_m:
            col1.metric("AUC - GLM", f"{glm_m['AUC']:.3f}")
            col2.metric("AUC - GBM", f"{gbm_m['AUC']:.3f}")
        if "Deviance" in glm_m and "LogLoss" in gbm_m:
            col1.metric("Deviancja (test) - GLM", f"{glm_m['Deviance']:.2f}")
            col2.metric("Log-loss (test) - GBM", f"{gbm_m['LogLoss']:.2f}")
        if "RMSE" in glm_m and "RMSE" in gbm_m:
            col1.metric("RMSE - GLM", f"{glm_m['RMSE']:.3f}")
            col2.metric("RMSE - GBM", f"{gbm_m['RMSE']:.3f}")

        # Wykresy kalibracji i krzywe Lorenza
        # Sprawdzamy, czy mamy przechowane predykcje na zbiorze testowym
        if 'y_test' in st.session_state:
            y_test = st.session_state.y_test
            glm_pred = st.session_state.get('glm_pred', None)
            gbm_pred = st.session_state.get('gbm_pred', None)
            # Jeśli target binarny i mamy prawdopodobieństwa
            if glm_pred is not None and gbm_pred is not None and len(y_test) == len(glm_pred) == len(gbm_pred):
                # Kalibracja: grupowanie w decyle i porównanie średnich
                calib_df = pd.DataFrame({
                    "y_true": y_test,
                    "glm_pred": glm_pred,
                    "gbm_pred": gbm_pred
                })
                calib_df = calib_df.sort_values("glm_pred")
                # 10 grup względem predykcji GLM (mogłoby być i względem GBM, ale wybierzmy jedną referencyjnie)
                calib_df['bin'] = pd.qcut(calib_df['glm_pred'], 10, labels=False, duplicates='drop')
                calib_results = calib_df.groupby('bin').agg({
                    'y_true': 'mean',
                    'glm_pred': 'mean',
                    'gbm_pred': 'mean'
                })
                # Tworzenie wykresu kalibracji
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Scatter(x=calib_results['glm_pred'], y=calib_results['y_true'],
                                             mode='lines+markers', name='GLM'))
                fig_cal.add_trace(go.Scatter(x=calib_results['gbm_pred'], y=calib_results['y_true'],
                                             mode='lines+markers', name='GBM'))
                # Linia idealnej kalibracji y=x
                fig_cal.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Idealna kalibracja', line=dict(dash='dash')))
                fig_cal.update_layout(title="Wykres kalibracji (decyle)", xaxis_title="Średnie przewidziane prawdopodobieństwo",
                                      yaxis_title="Rzeczywista częstość zdarzeń")
                st.plotly_chart(fig_cal, use_container_width=True)
                # Krzywe Lorenza
                # Sortujemy dane testowe malejąco wg przewidywanego ryzyka dla każdego modelu
                lorenz_df = pd.DataFrame({"y_true": y_test})
                lorenz_df['glm_pred'] = glm_pred
                lorenz_df['gbm_pred'] = gbm_pred
                lorenz_df = lorenz_df.sort_values('glm_pred', ascending=False).reset_index(drop=True)
                # Skumulowany % populacji i skumulowany % sumy targetu
                total_target = lorenz_df['y_true'].sum()
                lorenz_df['cum_pop_perc'] = (np.arange(len(lorenz_df)) + 1) / len(lorenz_df)
                # Oblicz kumulację targetu według sortowania GLM
                lorenz_df['cum_target_glm'] = np.cumsum(lorenz_df['y_true']) / total_target if total_target != 0 else np.cumsum(lorenz_df['y_true'])
                # Dla GBM sortujemy oddzielnie
                lorenz_df2 = lorenz_df.sort_values('gbm_pred', ascending=False).reset_index(drop=True)
                lorenz_df2['cum_target_gbm'] = np.cumsum(lorenz_df2['y_true']) / total_target if total_target != 0 else np.cumsum(lorenz_df2['y_true'])
                # Tworzenie wykresu Lorenz
                fig_lorenz = go.Figure()
                fig_lorenz.add_trace(go.Scatter(x=lorenz_df['cum_pop_perc'], y=lorenz_df['cum_target_glm'],
                                                mode='lines', name='GLM'))
                fig_lorenz.add_trace(go.Scatter(x=lorenz_df2['cum_pop_perc'], y=lorenz_df2['cum_target_gbm'],
                                                mode='lines', name='GBM'))
                # Linia równości (idealnie losowy model)
                fig_lorenz.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Losowy', line=dict(dash='dash')))
                fig_lorenz.update_layout(title="Krzywe Lorenza (kumulacja targetu vs populacja)",
                                         xaxis_title="Odsetek populacji (posortowanej wg modelu)",
                                         yaxis_title="Odsetek skumulowanego targetu")
                st.plotly_chart(fig_lorenz, use_container_width=True)
                # Zachowanie figur do ewentualnego raportu
                st.session_state.fig_calibration = fig_cal
                st.session_state.fig_lorenz = fig_lorenz
    else:
        st.info("Aby porównać modele, wytrenuj oba powyższe modele (GLM i GBM).")
