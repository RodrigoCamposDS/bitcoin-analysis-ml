# 🔶 ₿ ==| extract_fft_features |============= ₿ ============== | extract_fft_features |============== ₿ ===============| extract_fft_features |============ ₿ ============| extract_fft_features |============ ₿ =============| extract_fft_features |=====

import numpy as np
import pandas as pd
from hurst import compute_Hc

# =======================
# FUNÇÕES AUXILIARES
# =======================

def compute_hurst(series: np.ndarray) -> float:
    """Coeficiente de Hurst (memória longa) / Hurst exponent (long memory)"""
    series = series[~np.isnan(series)]
    H, _, _ = compute_Hc(series, kind='price')
    return H

def count_median_crossings(series: np.ndarray) -> int:
    """Número de cruzamentos com a mediana / Number of crossings with median"""
    median_val = np.median(series)
    shifted = (series - median_val) > 0
    return np.sum(shifted[1:] != shifted[:-1])

def longest_flat_spot(series: np.ndarray, tol=1e-6) -> int:
    """Maior sequência de valores quase constantes / Longest low-variation segment"""
    diff = np.abs(np.diff(series))
    flat = diff < tol
    max_len = count = 0
    for val in flat:
        count = count + 1 if val else 0
        max_len = max(max_len, count)
    return max_len

# =======================
# FUNÇÃO PRINCIPAL
# =======================

def extract_fft_features(series: pd.Series, top_k: int = 3) -> dict:
    """
    Extrai features estruturais da série temporal usando FFT + persistência e estagnação.
    / Extracts structural features using FFT + long memory and stagnation indicators.

    Parâmetros / Parameters:
    - series: pd.Series
        Série temporal com índice temporal / Time-indexed numeric series.
    - top_k: int
        Número de componentes principais de frequência / Number of dominant frequency components.

    Retorna / Returns:
    - dict com features de frequência + estrutura temporal /
      Dictionary with frequency + temporal structure features.
    """

    # =======================
    # PRÉ-PROCESSAMENTO
    # =======================
    series = series.dropna().values
    n = len(series)

    # =======================
    # TRANSFORMADA DE FOURIER
    # =======================
    fft_vals = np.fft.fft(series)
    fft_freqs = np.fft.fftfreq(n)

    pos_mask = fft_freqs > 0
    freqs = fft_freqs[pos_mask]
    magnitudes = np.abs(fft_vals[pos_mask])

    # =======================
    # FEATURES ESPECTRAIS
    # =======================
    dominant_freq = freqs[np.argmax(magnitudes)]
    total_energy = np.sum(magnitudes ** 2)
    topk_energy = np.sum(np.sort(magnitudes ** 2)[-top_k:])
    energy_ratio = topk_energy / total_energy

    prob_dist = magnitudes / np.sum(magnitudes)
    entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-12))

    # =======================
    # FEATURES TEMPORAIS ADICIONAIS
    # =======================
    hurst = compute_hurst(series)
    median_cross = count_median_crossings(series)
    flat_spot = longest_flat_spot(series)

    # =======================
    # RETORNO DAS FEATURES
    # =======================
    return {
        "fft_dominant_freq": dominant_freq,
        "fft_energy_ratio": energy_ratio,
        "fft_peak_amplitude": np.max(magnitudes),
        "fft_spectral_entropy": entropy,
        "hurst_exponent": hurst,
        "median_crossings": median_cross,
        "longest_flat_spot": flat_spot
    }


# 🔶 ₿ ==| detect_price_peaks |============= ₿ ============== | detect_price_peaks |============== ₿ ===============| detect_price_peaks |============ ₿ ============| detect_price_peaks |============ ₿ =============| detect_price_peaks |=====

import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# ================================================================
# EXTRAÇÃO DE PICOS E VALES EM SÉRIE TEMPORAL
# PEAK AND VALLEY DETECTION IN TIME SERIES
# ================================================================
def extract_peak_features(
    df: pd.DataFrame,
    column: str = "btc_price_usd",
    distance: int = 5
) -> pd.DataFrame:
    """
    Detecta picos e vales na série de preços e retorna apenas os pontos relevantes com metadados.
    / Detects peaks and valleys in the price series and returns flagged points with metadata.

    Parâmetros / Parameters:
    - df: pd.DataFrame
        DataFrame contendo 'btc_price_usd', 'block_height', 'block_timestamp' / DataFrame with price and metadata.
    - column: str
        Nome da coluna de preços / Name of the price column.
    - distance: int
        Distância mínima entre pontos / Minimum distance between detected events.

    Retorna / Returns:
    - pd.DataFrame com colunas: block_height, block_timestamp, is_peak, is_valley
    / DataFrame with event flags and tracking metadata.
    """
    # Verifica se a coluna 'block_timestamp' existe
    if 'block_timestamp' not in df.columns:
        print("Coluna 'block_timestamp' não encontrada, criando uma coluna fictícia.")
        df['block_timestamp'] = pd.date_range(start='2024-01-01', periods=len(df))

    if 'block_height' not in df.columns:
        print("Coluna 'block_height' não encontrada, criando uma coluna fictícia.")
        df['block_height'] = np.arange(len(df))

    df = df[[column, "block_height", "block_timestamp"]].copy()

    # ================================
    # PRÉ-PROCESSAMENTO / PREPROCESSING
    # ================================
    series = df[column].ffill().values
    # --> Preenche valores ausentes / Fill missing values <--

    # ================================
    # DETECÇÃO DE PONTOS / POINT DETECTION
    # ================================
    peaks, _ = find_peaks(series, distance=distance)
    valleys, _ = find_peaks(-series, distance=distance)
    # --> Usa find_peaks para detectar máximos e mínimos locais / Uses find_peaks for local maxima/minima <--

    # ================================
    # CRIA DATAFRAME DE RESULTADOS
    # CREATE RESULTING DATAFRAME
    # ================================
    peak_df = df.iloc[peaks].copy()
    peak_df["is_peak"] = 1
    peak_df["is_valley"] = 0

    valley_df = df.iloc[valleys].copy()
    valley_df["is_peak"] = 0
    valley_df["is_valley"] = 1

    result = pd.concat([peak_df, valley_df], axis=0).sort_index()
    # --> Junta picos e vales em um único DataFrame / Combine peaks and valleys <--

    return result[["block_height", "block_timestamp", "is_peak", "is_valley"]]
    # --> Retorna apenas colunas relevantes / Return only relevant columns <--


# 🔶 ₿ ==| extract_cycle_features |============= ₿ ============== | extract_cycle_features |============== ₿ ===============| extract_cycle_features |============ ₿ ============| extract_cycle_features |============ ₿ =============| extract_cycle_features |=====


import pandas as pd

# ================================================================
# EXTRAÇÃO DE FEATURES DE CICLOS BASEADOS EM VALES DETECTADOS
# EXTRACTS CYCLE-BASED FEATURES USING DETECTED VALLEYS
# ================================================================
def extract_cycle_features(df_peaks: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai estatísticas de ciclos com base nos vales detectados na série temporal.
    / Extracts cycle-based statistics using detected valleys in the time series.

    Parâmetros / Parameters:
    - df_peaks: pd.DataFrame
        DataFrame contendo colunas 'is_valley' e 'btc_price_usd'.
        / DataFrame with 'is_valley' and 'btc_price_usd' columns.

    Retorna / Returns:
    - pd.DataFrame
        DataFrame com colunas adicionais representando estatísticas dos ciclos.
        / DataFrame with additional columns for cycle-based statistics.
    """
    df = df_peaks.copy()

    # ================================
    # DEFINIÇÃO DOS CICLOS / CYCLE DELIMITATION
    # ================================
    df["cycle_id"] = (df["is_valley"] == 1).cumsum()
    # --> Cada vale define o início de um novo ciclo / Each valley starts a new cycle <--

    # ================================
    # ESTATÍSTICAS DOS CICLOS / CYCLE STATISTICS
    # ================================
    cycle_stats = df.groupby("cycle_id")["btc_price_usd"].agg([
        ("cycle_duration", "count"),
        # --> Duração do ciclo em pontos / Duration of each cycle in time steps <--

        ("cycle_amplitude", lambda x: x.max() - x.min())
        # --> Amplitude do ciclo (máx - mín) / Amplitude of the cycle (max - min) <--
    ]).reset_index()

    # ================================
    # MERGE DAS FEATURES / MERGE FEATURES INTO MAIN DF
    # ================================
    df = df.merge(cycle_stats, on="cycle_id", how="left")
    # --> Junta estatísticas de cada ciclo ao DataFrame principal / Merges cycle features into main DataFrame <--

    # ================================
    # TEMPO DESDE O ÚLTIMO VALE / TIME SINCE LAST VALLEY
    # ================================
    df["days_since_last_valley"] = df.groupby("cycle_id").cumcount()
    # --> Conta quantos pontos se passaram desde o último vale / Counts how many time steps since last valley <--

    return df

# 🔶 ₿ ==| reconstruct_fft |============= ₿ ============== | reconstruct_fft |============== ₿ ===============| reconstruct_fft |============ ₿ ============| reconstruct_fft |============ ₿ =============| reconstruct_fft |=====



def reconstruct_fft(series: np.ndarray, top_k: int = 3) -> np.ndarray:
    """
    Reconstrói a série temporal com base nas top-k frequências via FFT.
    / Reconstructs the time series using the top-k dominant frequencies via FFT.
    """
    series = series.dropna().values
    n = len(series)
    fft_vals = np.fft.fft(series)
    fft_power = np.abs(fft_vals)**2

    indices_topk = np.argsort(fft_power[1:n//2])[-top_k:] + 1
    fft_filtered = np.zeros_like(fft_vals, dtype=complex)

    fft_filtered[indices_topk] = fft_vals[indices_topk]
    fft_filtered[-indices_topk] = fft_vals[-indices_topk]  # conjugadas

    reconstructed = np.fft.ifft(fft_filtered).real
    return reconstructed


# 🔶 ₿ ==| prepare_fft_data |============= ₿ ============== | prepare_fft_data |============== ₿ ===============| prepare_fft_data |============ ₿ ============| prepare_fft_data |============ ₿ =============| prepare_fft_data |=====


import numpy as np
import pandas as pd

# ================================================================
# PREPARAÇÃO DA SÉRIE PARA FFT / PREPARE SERIES FOR FFT
# ================================================================
def prepare_fft_data(series: pd.Series):
    """
    Pré-processa a série e retorna as frequências e amplitudes positivas.
    / Preprocesses the series and returns positive frequencies and magnitudes.
    """
    series = series.dropna().values
    n = len(series)
    fft_vals = np.fft.fft(series)
    fft_freqs = np.fft.fftfreq(n)

    pos_mask = fft_freqs > 0
    freqs = fft_freqs[pos_mask]
    magnitudes = np.abs(fft_vals[pos_mask])

    return freqs, magnitudes


# 🔶 ₿ ==| calculate_energy_ratio |============= ₿ ============== | calculate_energy_ratio |============== ₿ ===============| calculate_energy_ratio |============ ₿ ============| calculate_energy_ratio |============ ₿ =============| calculate_energy_ratio |=====


# ================================================================
# CÁLCULO DA RAZÃO DE ENERGIA / ENERGY RATIO
# ================================================================
def calculate_energy_ratio(magnitudes: np.ndarray, top_k: int):
    """
    Calcula a razão entre a energia das top-k amplitudes e a energia total.
    / Calculates the ratio between top-k dominant energy and total energy.
    """
    total_energy = np.sum(magnitudes**2)
    topk_energy = np.sum(np.sort(magnitudes**2)[-top_k:])
    return topk_energy / total_energy


# 🔶 ₿ ==| calculate_spectral_entropy |============= ₿ ============== | calculate_spectral_entropy |============== ₿ ===============| calculate_spectral_entropy |============ ₿ ============| calculate_spectral_entropy |============ ₿ =============| calculate_spectral_entropy |=====


# ================================================================
# CÁLCULO DA ENTROPIA ESPECTRAL / SPECTRAL ENTROPY
# ================================================================
def calculate_spectral_entropy(magnitudes: np.ndarray):
    """
    Calcula a entropia espectral da distribuição de frequência.
    / Computes the spectral entropy of the frequency distribution.
    """
    prob_dist = magnitudes / np.sum(magnitudes)
    return -np.sum(prob_dist * np.log2(prob_dist + 1e-12))


# 🔶 ₿ ==| extract_cyclic_features |============= ₿ ============== | extract_cyclic_features |============== ₿ ===============| extract_cyclic_features |============ ₿ ============| extract_cyclic_features |============ ₿ =============| extract_cyclic_features |=====


import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def extract_cyclic_features(
    series: pd.Series,
    block_height: int,
    block_timestamp: pd.Timestamp,
    distance: int = 10
) -> pd.DataFrame:
    """
    Extrai features cíclicas com base em picos e vales.
    / Extracts cycle-related features using peaks and troughs.

    Parâmetros / Parameters:
    - series: pd.Series
        Série temporal com valores numéricos / Numeric time series.
    - block_height: int
        Altura do bloco para rastreabilidade / Block height for traceability.
    - block_timestamp: pd.Timestamp
        Timestamp da última observação da janela / Final timestamp of the window.
    - distance: int
        Distância mínima entre picos/vales / Minimum distance between peaks/troughs.

    Retorna / Returns:
    - pd.DataFrame com uma linha contendo as features cíclicas / One-row DataFrame with cycle features.
    """
    series = series.reset_index(drop=True).dropna()
    valores = series.values

    peaks, _ = find_peaks(valores, distance=distance)
    troughs, _ = find_peaks(-valores, distance=distance)

    if len(troughs) < 2 or len(peaks) < 1:
        return pd.DataFrame([{
            "block_height": block_height,
            "block_timestamp": block_timestamp,
            "dias_desde_ultimo_vale": np.nan,
            "duracao_ultimo_ciclo": np.nan,
            "amplitude_ultimo_ciclo": np.nan,
            "fase_ciclo": np.nan,
            "flag_em_topo": 0
        }])

    ultimo_vale = troughs[-1]
    penultimo_vale = troughs[-2]
    pico_central = [p for p in peaks if penultimo_vale < p < ultimo_vale]

    if not pico_central:
        return pd.DataFrame([{
            "block_height": block_height,
            "block_timestamp": block_timestamp,
            "dias_desde_ultimo_vale": len(series) - ultimo_vale,
            "duracao_ultimo_ciclo": ultimo_vale - penultimo_vale,
            "amplitude_ultimo_ciclo": np.nan,
            "fase_ciclo": np.nan,
            "flag_em_topo": 0
        }])

    pico = pico_central[-1]
    amplitude = valores[pico] - valores[penultimo_vale]
    duracao = ultimo_vale - penultimo_vale
    dias_desde_vale = len(series) - ultimo_vale
    fase_ciclo = dias_desde_vale / duracao if duracao > 0 else np.nan

    return pd.DataFrame([{
        "block_height": block_height,
        "block_timestamp": block_timestamp,
        "dias_desde_ultimo_vale": dias_desde_vale,
        "duracao_ultimo_ciclo": duracao,
        "amplitude_ultimo_ciclo": amplitude,
        "fase_ciclo": fase_ciclo,
        "flag_em_topo": int(peaks[-1] == len(series) - 1)
    }])