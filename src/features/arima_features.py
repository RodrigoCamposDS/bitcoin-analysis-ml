import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# ================================================================
# EXTRAÇÃO DE FEATURES ARIMA / ARIMA FEATURE EXTRACTION
# ================================================================
def extract_arima_features(
    series: pd.Series,
    block_height: int,
    block_timestamp: pd.Timestamp,
    order=(1, 1, 1),
    window_size: int = 60,
    step: int = 10
) -> pd.DataFrame:
    """
    Ajusta um modelo ARIMA e retorna valores ajustados e resíduos com rastreamento para várias janelas.
    / Fits an ARIMA model and returns fitted values and residuals with tracking for multiple windows.

    Parâmetros / Parameters:
    - series: pd.Series com índice datetime e valores numéricos / Time-indexed numeric series
    - block_height: int
        Altura do bloco da série analisada / Block height for traceability
    - block_timestamp: pd.Timestamp
        Timestamp associado ao final da janela / Timestamp of the window
    - order: tuple (p, d, q) com os parâmetros ARIMA / ARIMA model order (p, d, q)
    - window_size: int
        Tamanho da janela para a análise / Window size for analysis
    - step: int
        Passo entre janelas / Step between windows

    Retorna / Returns:
    - pd.DataFrame com ['block_height', 'block_timestamp', 'arima_fitted', 'arima_resid']
    / DataFrame with tracking info, fitted values and residuals
    """

    results = []  # Lista para armazenar os resultados das janelas / List to store window results

    # Loop de extração para várias janelas / Loop for multiple windows
    for i in range(0, len(series) - window_size + 1, step):
        sub_series = series.iloc[i:i + window_size]
        # --> Subconjunto da série para a janela atual / Subset of the series for the current window

        block_height_window = block_height
        block_timestamp_window = block_timestamp
        # --> Chaves de rastreio estáticas associadas à janela / Static tracking keys for the window <--

        # ================================
        # VERIFICAÇÃO DE TAMANHO MÍNIMO / MINIMUM SIZE CHECK
        # ================================
        if sub_series.dropna().shape[0] < max(order) + 2:
            print(f"[AVISO] Série muito curta na janela {i}:{i + window_size}. Ignorando.")
            continue
            # --> Série muito curta para ajuste ARIMA / Too short to fit ARIMA <--

        try:
            # ================================
            # AJUSTE DO MODELO ARIMA / ARIMA FITTING
            # ================================
            model = ARIMA(sub_series, order=order)
            model_fit = model.fit()
            # --> Ajusta o modelo ARIMA com a janela atual / Fit ARIMA model on current window <--

            fitted_values = model_fit.fittedvalues
            residuals = model_fit.resid
            # --> Obtém valores ajustados e resíduos / Get fitted values and residuals <--

            result = pd.DataFrame({
                "block_height": block_height_window,
                "block_timestamp": block_timestamp_window,
                "arima_fitted": fitted_values,
                "arima_resid": residuals
            }, index=sub_series.index)
            # --> Formata o resultado como DataFrame rastreável / Format result with tracking index <--

        except Exception as e:
            print(f"[ERRO] Falha no ajuste ARIMA na janela {i}:{i + window_size} — {e}")
            result = pd.DataFrame({
                "block_height": [block_height_window] * len(sub_series),
                "block_timestamp": [block_timestamp_window] * len(sub_series),
                "arima_fitted": [np.nan] * len(sub_series),
                "arima_resid": [np.nan] * len(sub_series)
            }, index=sub_series.index)
            # --> Em caso de erro, retorna linha com NaNs / On error, fallback to NaNs <--

        results.append(result)

    # ================================
    # CONSOLIDAÇÃO FINAL / FINAL CONCATENATION
    # ================================
    if results:
        df_arima = pd.concat(results, axis=0)
        df_arima = df_arima.dropna(subset=["arima_fitted", "arima_resid"])
        # --> Remove linhas com NaNs causadas por diferenciação ou falhas / Drop rows with NaNs <--
        return df_arima
    else:
        print("[AVISO] Nenhuma janela ARIMA válida foi processada.")
        return pd.DataFrame()
        # --> Retorna DataFrame vazio caso nenhuma janela válida / Return empty DataFrame if no valid window <--