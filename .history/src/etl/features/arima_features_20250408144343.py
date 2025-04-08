import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

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
        sub_series = series[i:i + window_size]
        # --> Subconjunto da série para a janela atual / Subset of the series for the current window
        block_height_window = block_height  # Chave de rastreio do bloco / Block height key
        block_timestamp_window = block_timestamp  # Chave de rastreio do timestamp / Block timestamp key

        # Ajuste do modelo ARIMA / ARIMA model fit
        model = ARIMA(sub_series, order=order)
        model_fit = model.fit()

        fitted_values = model_fit.fittedvalues
        residuals = model_fit.resid

        # Armazenamento do resultado para a janela / Store the result for the window
        result = pd.DataFrame({
            "block_height": block_height_window,
            "block_timestamp": block_timestamp_window,
            "arima_fitted": fitted_values,
            "arima_resid": residuals
        }, index=sub_series.index)

        results.append(result)

    # Concatenar os resultados das janelas / Concatenate the results for all windows
    df_arima = pd.concat(results, axis=0)

    return df_arima.dropna()
    # --> Remove linhas com NaNs (por causa da diferenciação) / Drop NaNs from differencing