import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ================================================================
# EXTRAÇÃO DE FEATURES ESTRUTURAIS VIA STL / STL STRUCTURAL FEATURES
# ================================================================
def extract_stl_features(
    series: pd.Series,
    block_height: int,
    block_timestamp: pd.Timestamp,
    period: int = 48
) -> pd.DataFrame:
    """
    Aplica decomposição STL e extrai métricas estruturais.
    / Applies STL decomposition and extracts structural time series metrics.

    Parâmetros / Parameters:
    - series: pd.Series
        Série de preços ou variável alvo / Target time series.
    - block_height: int
        Altura do bloco de referência / Reference block height.
    - block_timestamp: pd.Timestamp
        Timestamp de referência para a janela analisada / Reference timestamp.
    - period: int
        Periodicidade da sazonalidade (ex: 48 = 1 dia de 30min) / Seasonal period.

    Retorna / Returns:
    - pd.DataFrame com uma linha e colunas de features estruturais / One-row DataFrame with STL structural features.
    """

    # ================================
    # PREPARAÇÃO DA SÉRIE / PREPROCESSING
    # ================================
    series = series.interpolate().dropna()
    series.index = pd.to_datetime(series.index)
    # --> Interpola, remove NaNs e converte índice para datetime / Interpolates, drops NaNs and enforces datetime index <--

    result = STL(series, period=period, robust=True).fit()
    # --> Ajusta a decomposição STL com robustez / Fits STL decomposition <--

    trend = result.trend
    seasonal = result.seasonal
    resid = result.resid
    # --> Extrai os componentes STL / Extract STL components <--

    # ================================
    # MÉDIA SAZONAL POR MÊS / SEASONAL PROFILE
    # ================================
    seasonal_df = pd.DataFrame({
        "seasonal": seasonal,
        "month": series.index.month
    })
    seasonal_mean = seasonal_df.groupby("month")["seasonal"].mean()
    seasonal_peak_month = seasonal_mean.idxmax()
    seasonal_trough_month = seasonal_mean.idxmin()
    # --> Mês com maior e menor componente sazonal / Peak and trough month of seasonal component <--

    # ================================
    # SPIKINESS DOS RESÍDUOS
    # ================================
    leave_one_out_vars = [np.var(np.delete(resid.values, i)) for i in range(len(resid))]
    spikiness = np.var(leave_one_out_vars)
    # --> Variância leave-one-out como medida de spikiness / Leave-one-out variance <--

    # ================================
    # LINEARIDADE E CURVATURA DA TENDÊNCIA
    # ================================
    t = np.arange(len(trend)).reshape(-1, 1)
    linearity = LinearRegression().fit(t, trend).coef_[0]
    # --> Coeficiente linear / Linear coefficient <--

    poly = PolynomialFeatures(degree=2, include_bias=False)
    curvature = LinearRegression().fit(poly.fit_transform(t), trend).coef_[1]
    # --> Coeficiente de curvatura (t²) / Curvature coefficient <--

    # ================================
    # AUTOCORRELAÇÃO DOS RESÍDUOS
    # ================================
    acf_vals = acf(resid, nlags=10, fft=False)
    stl_e_acf1 = acf_vals[1]
    stl_e_acf10 = np.sum(acf_vals[1:11] ** 2)
    # --> ACF 1 e soma dos quadrados até lag 10 / ACF 1 and ACF^2 sum <--

    # ================================
    # DATAFRAME FINAL
    # ================================
    return pd.DataFrame([{
        "block_height": block_height,
        "block_timestamp": block_timestamp,
        "seasonal_peak_month": seasonal_peak_month,
        "seasonal_trough_month": seasonal_trough_month,
        "stl_spikiness": spikiness,
        "stl_trend_linearity": linearity,
        "stl_trend_curvature": curvature,
        "stl_e_acf1": stl_e_acf1,
        "stl_e_acf10": stl_e_acf10
    }])
    # --> Retorna features em formato de linha única com chaves de rastreio / Return row-wise feature set with tracking keys <--