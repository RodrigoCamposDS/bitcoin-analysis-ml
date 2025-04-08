import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# ================================================================
# EXTRAÇÃO DE MUDANÇAS DE REGIME VIA ESTATÍSTICAS ROLLING
# REGIME SHIFT FEATURES USING ROLLING ADF AND VOLATILITY
# ================================================================
def extract_regime_features(
    df: pd.DataFrame,
    col: str = "btc_price_usd",
    window: int = 20
) -> pd.DataFrame:
    """
    Extrai features de mudança de regime e variabilidade em janelas móveis.
    / Extracts regime shift and local variability features using rolling ADF and volatility.

    Parâmetros / Parameters:
    - df: pd.DataFrame
        DataFrame contendo colunas 'block_height', 'block_timestamp' e série alvo / DataFrame with price and metadata
    - col: str
        Nome da coluna de preços / Name of the column to analyze
    - window: int
        Tamanho da janela de análise / Rolling window size

    Retorna / Returns:
    - pd.DataFrame com colunas: block_height, block_timestamp, rolling_mean, rolling_std, stationarity_zscore, is_regime_change
    """

    # -------------------------------
    # Cópia e validação dos dados / Copy and prepare
    # -------------------------------
    df = df[[col, "block_height", "block_timestamp"]].copy()

    # -------------------------------
    # MÉDIA E VOLATILIDADE MÓVEL
    # ROLLING MEAN AND STD DEV
    # -------------------------------
    df["rolling_mean"] = df[col].rolling(window=window).mean()
    df["rolling_std"] = df[col].rolling(window=window).std()
    # --> Calcula tendência e dispersão local / Compute local trend and dispersion <--

    # -------------------------------
    # Z-SCORE DA VARIÂNCIA LOCAL
    # LOCAL VARIANCE Z-SCORE
    # -------------------------------
    rolling_var = df[col].rolling(window=window).var()
    global_var = df[col].var()
    df["stationarity_zscore"] = (rolling_var - global_var) / (global_var + 1e-8)
    # --> Compara variância local com global / Compare local variance to global <--

    # -------------------------------
    # FLAG DE MUDANÇA DE REGIME VIA ADF
    # REGIME SHIFT FLAG USING ADF TEST
    # -------------------------------
    regime_flags = []
    for i in range(len(df)):
        if i < window:
            regime_flags.append(np.nan)
        else:
            sub_series = df[col].iloc[i - window:i].dropna()
            if len(sub_series) < window:
                regime_flags.append(np.nan)
            else:
                try:
                    p_val = adfuller(sub_series, autolag='AIC')[1]
                    regime_flags.append(int(p_val > 0.05))  # p > 0.05 → não estacionária / non-stationary
                except:
                    regime_flags.append(np.nan)

    df["is_regime_change"] = regime_flags
    df["is_regime_change"] = df["is_regime_change"].fillna(0).astype(int)
    # --> 1 indica possível mudança de regime / 1 = possible regime change <--

    # -------------------------------
    # RETORNO FINAL
    # FINAL OUTPUT
    # -------------------------------
    return df[[
        "block_height", "block_timestamp",
        "rolling_mean", "rolling_std",
        "stationarity_zscore", "is_regime_change"
    ]]