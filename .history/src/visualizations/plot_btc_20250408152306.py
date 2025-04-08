# 🔶 ₿ ==| plot_btc_price_timeseries |============= ₿ ============== | plot_btc_price_timeseries |============== ₿ ===============| plot_btc_price_timeseries |============ ₿ ============| plot_btc_price_timeseries |============ ₿ =============| plot_btc_price_timeseries |=====
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_btc_price_timeseries(df_graph, image_path,resample ):
    """
    Gera um gráfico de linha com a série temporal do preço do Bitcoin, preenchido até o eixo X,
    com marca d'água do ícone do BTC.

    Parâmetros:
    - df_graph (pd.DataFrame): DataFrame com a coluna 'btc_price_usd'.
    - image_path (str): Caminho para o arquivo de imagem a ser usado como marca d'água.
    """
    df_graph = df_graph.resample(resample).mean()
    # --> Reamostra os dados com frequência diária, calculando a média por dia / Resamples data to daily frequency, taking mean per day <--

    df_graph = df_graph.reset_index()
    # --> Reseta o índice numérico do DataFrame, removendo o timestamp / Reset the DataFrame index, dropping timestamp column <--

    # Definir os valores mínimo e máximo automaticamente com um buffer de 2%
    y_min = df_graph["btc_price_usd"].min() * 0.98  # 2% abaixo do mínimo / 2% below min
    y_max = df_graph["btc_price_usd"].max() * 1.02  # 2% acima do máximo / 2% above max

    # Criar gráfico base
    fig = px.line(
        df_graph,
        x="block_timestamp" ,  # --> Usar o índice numérico como eixo X / Use numeric index as X-axis <--
        y="btc_price_usd",
        labels={"block_timestamp": "Tempo", "btc_price_usd": "Preço BTC"},
        template="plotly_dark"
    )

    fig.update_traces(
        line=dict(color="#E57C1F", width=1.5),  # --> Define a cor e largura da linha / Set line color and width <--
        fill='tozeroy',
        fillcolor="rgba(229, 165, 0, 0.1)",  # --> Preenchimento suave abaixo da linha / Light fill under the curve <--
        hovertemplate="%{x|%d/%m/%Y %H:%M}<br>Preço BTC: %{y:$,.2f}<extra></extra>"  
        # --> Tooltip formatada com data e valor / Formatted hover info with date and price <--
    )

    fig.update_xaxes(
        tickformat="%d/%m\n%H:%M",  # --> Formatação do eixo X (data e hora) / Date-time format for X-axis <--
        title="Data e Hora"
    )

    fig.update_layout(
        height=600,
        showlegend=False,
        title={
            "text": '<b><span style="font-size:22px;">Série Temporal do Preço do Bitcoin</span></b>',
            "x": 0.5, "y": 0.94,  # --> Centraliza e posiciona o título / Centers and lifts the title <--
            "xanchor": "center", "yanchor": "top"
        },
        yaxis=dict(
            title="Valor do Bitcoin - USD",
            side="left",
            range=[y_min, y_max]  # --> Define a escala dinâmica do eixo Y / Auto-scale Y with buffer <--
        ),
        xaxis=dict(
            title="Data"
        )
    )

    # ==========================
    # INSERIR IMAGEM NO GRÁFICO
    # ==========================

    fig.add_layout_image(
        dict(
            source=image_path,
            xref="paper", yref="paper",
            x=0.008, y=1.1,
            sizex=0.15, sizey=0.15,
            xanchor="left", yanchor="top"
        )
    )
    # --> Adiciona ícone do Bitcoin como marca d’água / Adds Bitcoin logo as visual watermark <--

    fig.show()
    # --> Exibe o gráfico / Display the chart <--


# 🔶 ₿ ==| plot_btc_boxplot_by_weekday |============= ₿ ============== | plot_btc_boxplot_by_weekday |============== ₿ ===============| plot_btc_boxplot_by_weekday |============ ₿ ============| plot_btc_boxplot_by_weekday |============ ₿ =============| plot_btc_boxplot_by_weekday |=====


import pandas as pd
import plotly.express as px

def plot_btc_boxplot_by_week(df, image_path):
    """
    Gera boxplot do Preço do Bitcoin por Semana do Ano (YYYY-WW).

    Parâmetros:
    - df: DataFrame com colunas ['block_timestamp', 'btc_price_usd']
    - image_path: caminho do arquivo da imagem do ícone Bitcoin
    """

    # ===========================
    # PREPARAÇÃO DOS DADOS
    # ===========================
    if "block_timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df["block_timestamp"] = df.index
        else:
            df = df.reset_index(drop=False)
    # --> Garante que 'block_timestamp' esteja presente como coluna / Ensures 'block_timestamp' is a column <--

    df["block_timestamp"] = pd.to_datetime(df["block_timestamp"])
    # --> Converte a coluna para datetime / Converts to datetime format <--

    # ===============================
    # AGRUPAMENTO POR SEMANA (YYYY-WW)
    # ===============================
    df["year_week"] = df["block_timestamp"].dt.strftime("%Y-W%U")
    # --> Cria coluna com ano e número da semana / Creates column with year-week format <--

    df = df.sort_values("year_week")
    # --> Ordena cronologicamente por semana / Sort by week for consistent plotting <--

    # ===============================
    # LIMITES DO EIXO Y COM BUFFER
    # ===============================
    y_min = df["btc_price_usd"].min() * 0.98
    y_max = df["btc_price_usd"].max() * 1.02
    # --> Define uma margem de 2% abaixo/acima para melhor visualização / Sets a 2% margin below/above for Y-axis <--

    # ===============================
    # BOXPLOT SEMANAL
    # ===============================
    fig = px.box(
        df,
        x="year_week",
        y="btc_price_usd",
        labels={"year_week": "Semana do Ano", "btc_price_usd": "Preço BTC"},
        template="plotly_dark"
    )
    # --> Cria gráfico boxplot agrupado por semana do ano / Weekly grouped boxplot <--

    fig.update_traces(
        line=dict(color="#E57C1F"),
        marker=dict(color="#E57C1F")
    )
    # --> Aplica cor laranja aos elementos visuais / Applies orange styling <--

    # ========================
    # LAYOUT DO GRÁFICO
    # ========================
    fig.update_layout(
        height=600,
        showlegend=False,
        title={
            "text": '<b><span style="font-size:22px;">Sazonalidade Semanal do Preço do Bitcoin</span></b>',
            "x": 0.5, "y": 0.94,
            "xanchor": "center", "yanchor": "top"
        },
        yaxis=dict(
            title="Valor do Bitcoin - USD",
            side="left",
            range=[y_min, y_max]
        ),
        xaxis=dict(
            title="Semana do Ano (YYYY-WW)",
            tickangle=45
        )
    )
    # --> Define layout com título centralizado e eixo X rotacionado / Layout settings <--

    # ============================
    # INSERE MARCA D'ÁGUA (BTC)
    # ============================
    fig.add_layout_image(
        dict(
            source=image_path,
            xref="paper", yref="paper",
            x=0.008, y=1.1,
            sizex=0.1, sizey=0.1,
            xanchor="left", yanchor="top"
        )
    )
    # --> Insere o ícone do Bitcoin no canto superior esquerdo / Adds watermark logo <--

    fig.show()
    # --> Exibe o gráfico / Displays the plot <--

# 🔶 ₿ ==| plot_histogram_variacao_btc |============= ₿ ============== | plot_histogram_variacao_btc |============== ₿ ===============| plot_histogram_variacao_btc |============ ₿ ============| plot_histogram_variacao_btc |============ ₿ =============| plot_histogram_variacao_btc |=====


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def plot_histogram_variacao_btc(df_graph, image_path):
    """
    Plota um histograma da variação percentual do Bitcoin com estilo visual customizado.

    Parameters:
    - df_graph: DataFrame contendo a coluna 'btc_price_usd'
    - image_path: Caminho para a imagem do Bitcoin que será adicionada ao gráfico

    Returns:
    - fig: objeto Plotly Figure com o gráfico gerado
    """

    # Criar histograma da coluna 'btc_price_usd'
    fig = px.histogram(
        df_graph,
        x="btc_price_usd",
        nbins=50,
        title="Distribuição da Variação Percentual do BTC",
        labels={"btc_price_usd": "Variação Percentual"},
        template="plotly_dark"
    )
    # --> Cria histograma da coluna 'btc_price_usd' com 50 bins, usando o tema escuro / Creates a histogram with 50 bins using dark theme <--

    fig.update_traces(
        marker=dict(color="rgba(229, 165, 0, 0.2)", line=dict(color="#E57C1F", width=1))
    )
    # --> Aplica cor laranja translúcida nas barras e contorno mais escuro / Applies translucent orange with sharp border <--

    # Layout do gráfico
    fig.update_layout(
        height=800,
        template="plotly_dark",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        title={
            "text": '<b><span style="font-size:22px;">Bitcoin BTC - Distribuição da Variação Percentual do BTC</span></b>',
            "x": 0.5, "y": 0.95,
            "xanchor": "center", "yanchor": "top"
        },
        bargap=0.05,
        xaxis_title="Preço do Bitcoin - USD",
        yaxis_title="Variação Percentual",
        yaxis=dict(title="Variação Percentual", side="left"),
    )
    # --> Define estilo visual: fundo preto, título centralizado, espaçamento entre barras e eixos nomeados / Sets visual theme and formatting <--

    fig.add_layout_image(
        dict(
            source=image_path,
            xref="paper", yref="paper",
            x=0.008, y=1.15,
            sizex=0.15, sizey=0.15,
            xanchor="left", yanchor="top",
        )
    )
    # --> Adiciona imagem decorativa do Bitcoin no canto superior esquerdo / Adds BTC logo to upper-left corner <--

   
    

    return fig


# 🔶 ₿ ==| plot_series_comparativa |============= ₿ ============== | plot_series_comparativa |============== ₿ ===============| plot_series_comparativa |============ ₿ ============| plot_series_comparativa |============ ₿ =============| plot_series_comparativa |=====


import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_series_comparativa(df_graph, df_graph_stationary, y_min, y_max):
    """
    Gera gráfico com duas séries temporais (original e diferenciada) para análise de estacionariedade.

    Parâmetros:
    - df_graph: DataFrame com a série original (com coluna 'block_timestamp' e 'btc_price_usd')
    - df_graph_stationary: DataFrame com a série diferenciada (com coluna 'btc_price_diff')
    - y_min: valor mínimo do eixo Y para a série original
    - y_max: valor máximo do eixo Y para a série original
    """    

    # ==========================
    # ESTRUTURA DE SUBPLOTS
    # ==========================
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Série Temporal do Preço do Bitcoin (Não Estacionária)",
            "Série Temporal Diferenciada (Estacionária Aproximadamente)"
        ]
    )
    # --> Cria estrutura de subplots com 2 linhas e 1 coluna / Creates subplot layout with 2 rows, 1 column <--

    # ==========================
    # GRÁFICO 1: Série Original
    # ==========================
    fig.add_trace(go.Scatter(
        x=df_graph["block_timestamp"],
        y=df_graph["btc_price_usd"],
        mode="lines",
        name="Preço BTC",
        line=dict(color="#E57C1F", width=1.5)
    ), row=1, col=1)
    # --> Adiciona linha da série original de preços do Bitcoin / Adds original BTC price series line <--

    # ===============================================
    # GRÁFICO 2: Série Diferenciada (Estacionária)
    # ===============================================
    fig.add_trace(go.Scatter(
        x=df_graph_stationary["block_timestamp"],
        y=df_graph_stationary["btc_price_diff"],
        mode="lines",
        name="Δ Preço BTC",
        line=dict(color="#1FADE5", width=1.5)
    ), row=2, col=1)
    # --> Adiciona linha da série diferenciada (primeira ordem) / Adds differenced series line (first-order) <--

    # ===========================
    # LAYOUT E ESTILO DO GRÁFICO
    # ===========================
    fig.update_layout(
        height=800,
        template="plotly_dark",
        showlegend=False,
        title={
            "text": "<b>Série Temporal do Bitcoin: Original vs Estacionária</b>",
            "x": 0.5,  # --> Centraliza o título / Centers the title <--
            "y": 0.95,
            "xanchor": "center",
            "yanchor": "top"
        }
    )

    fig.update_xaxes(
        tickformat="%d/%m\n%Y",
        title="Data",
        row=2, col=1
    )
    # --> Configura o formato da data no eixo X / Sets date format on X-axis <--

    fig.update_yaxes(title="Valor do Bitcoin - USD", range=[y_min, y_max], row=1, col=1)
    fig.update_yaxes(title="Δ Valor do Bitcoin", row=2, col=1)
    # --> Define os títulos e escalas dos eixos Y para cada subplot / Sets Y-axis titles and ranges for each subplot <--

    fig.show()
    # --> Exibe o gráfico interativo / Displays the interactive chart <--



# 🔶 ₿ ==| plot_acf_diferenciada |============= ₿ ============== | plot_acf_diferenciada |============== ₿ ===============| plot_acf_diferenciada |============ ₿ ============| plot_acf_diferenciada |============ ₿ =============| plot_acf_diferenciada |=====


import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf

def plot_acf_diferenciada(serie_diff, nlags, nlags_option):
    """
    Plota a ACF (Autocorrelação) de uma série temporal diferenciada com limites de confiança.

    Parâmetros:
    - serie_diff (pd.Series): Série diferenciada (Y(t) = Z(t) - Z(t-1))
    - nlags (int): Número de defasagens (lags) a considerar na ACF
    - nlags_option (str): Descrição da escolha do lag (ex: 'manual', 'auto', 'serie_longa')

    Retorna:
    - fig: Objeto Plotly com o gráfico ACF
    """

    # =====================
    # CÁLCULO DA ACF
    # =====================
    acf_vals = acf(serie_diff, nlags=nlags)
    # --> Calcula a função de autocorrelação (ACF) até o número de lags escolhido / Calculates the ACF up to chosen lag <--

    lags = list(range(1, len(acf_vals)))
    acf_vals_filtered = acf_vals[1:]
    # --> Remove o lag 0 (autocorrelação perfeita) da visualização / Removes lag 0 (perfect autocorrelation) <--

    conf = 1.96 / np.sqrt(len(serie_diff))
    # --> Calcula os limites de confiança de 95% / Computes 95% confidence bounds <--

    # =====================
    # PLOTAGEM DO GRÁFICO
    # =====================
    fig = go.Figure()
    # --> Inicializa o gráfico com Plotly / Initializes the Plotly figure <--

    fig.add_trace(go.Bar(
        x=lags,
        y=acf_vals_filtered,
        marker_color="#E57C1F",
        name="ACF"
    ))
    # --> Adiciona as barras de autocorrelação / Adds ACF bars to the plot <--

    # Linhas de confiança
    fig.add_shape(type="line", x0=1, x1=nlags, y0=conf, y1=conf,
                  line=dict(color="blue", dash="dash"))
    fig.add_shape(type="line", x0=1, x1=nlags, y0=-conf, y1=-conf,
                  line=dict(color="blue", dash="dash"))
    # --> Adiciona linhas de confiança superior e inferior / Adds confidence interval lines <--

    # Layout
    fig.update_layout(
        template="plotly_dark",
        title=f"ACF da Série Diferenciada — Lag: {nlags_option} ({nlags})",
        xaxis_title="Defasagem (Lag)",
        yaxis_title="Autocorrelação",
        height=500
    )
    # --> Define estilo visual, títulos e layout / Sets layout and visual theme <--

    return fig


# 🔶 ₿ ==| plot_rolling_mean_std |============= ₿ ============== | plot_rolling_mean_std |============== ₿ ===============| plot_rolling_mean_std |============ ₿ ============| plot_rolling_mean_std |============ ₿ =============| plot_rolling_mean_std |=====


import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_rolling_mean_std(df_Rolling, rolling_mean, rolling_std):
    """
    Gera um gráfico com a média móvel e o desvio padrão da série diferenciada do Bitcoin.

    Parameters:
    - df_Rolling (DataFrame): DataFrame com a coluna 'block_timestamp' e 'btc_price_diff'.
    - rolling_mean (Series): Série com a média móvel calculada.
    - rolling_std (Series): Série com o desvio padrão móvel calculado.

    Returns:
    - Exibe o gráfico interativo.
    """

    # ==========  PLOTAGEM  ==========
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("Rolling Mean e Std — Diagnóstico de Estacionariedade",),
        vertical_spacing=1  # --> Espaço vertical entre subplots (ainda que só haja um) / Vertical spacing (even with one subplot) <--
    )

    fig.add_trace(go.Scatter(
        x=df_Rolling["block_timestamp"],
        y=df_Rolling["btc_price_diff"],
        mode="lines",
        name="Δ Preço BTC",
        line=dict(color="#E57C1F", width=1.5)
    ), row=1, col=1)
    # --> Linha principal com a série diferenciada do preço do BTC / Main line with differenced BTC price series <--

    fig.add_trace(go.Scatter(
        x=df_Rolling["block_timestamp"],
        y=rolling_mean,
        mode="lines",
        name="Média Móvel",
        line=dict(color="blue", width=2)
    ), row=1, col=1)
    # --> Adiciona a média móvel (rolling mean) da série / Adds rolling mean to the plot <--

    fig.add_trace(go.Scatter(
        x=df_Rolling["block_timestamp"],
        y=rolling_std,
        mode="lines",
        name="Desvio Padrão Móvel",
        line=dict(color="#1FADE5", width=2)
    ), row=1, col=1)
    # --> Adiciona o desvio padrão móvel (rolling std) da série / Adds rolling standard deviation to the plot <--

    fig.update_layout(
        height=500,
        template="plotly_dark",
        title={
            "text": "<b>Rolling Mean & Std da Série Diferenciada</b>",
            "x": 0.5,            # --> Centraliza horizontalmente o título / Horizontally centers the title <--
            "y": 0.85,           # --> Ajusta altura do título / Controls vertical position of title <--
            "xanchor": "center",
            "yanchor": "top"
        },
        legend=dict(
            y=1.5,               # --> Posiciona a legenda acima do gráfico / Moves legend above the plot <--
            x=0.99,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",  # --> Fundo transparente da legenda / Transparent legend background <--
            bordercolor="gray",
            borderwidth=1
        ),
        xaxis_title="Data",
        yaxis_title="Valor"
        # --> Títulos dos eixos X e Y / Axis titles <--
    )

    fig.show()
    # --> Exibe o gráfico com média móvel e desvio móvel sobrepostos à série diferenciada / Displays the final chart with rolling mean/std <--


# 🔶 ₿ ==| plot_acf_pacf |============= ₿ ============== | plot_acf_pacf |============== ₿ ===============| plot_acf_pacf |============ ₿ ============| plot_acf_pacf |============ ₿ =============| plot_acf_pacf |=====


import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf, pacf


def plot_acf_pacf(df_features_temp, resample_freq='h', nlags_option='serie_longa', manual_nlags=150, 
                  date_range_start=None, date_range_end=None):
    # =======================
    # CONFIGURAÇÕES INICIAIS
    # =======================

    df_acf = df_features_temp.toPandas()
    # --> Converte o DataFrame do PySpark para Pandas / Converts PySpark DataFrame to Pandas <--

    df_acf["block_timestamp"] = pd.to_datetime(df_acf["block_timestamp"])
    # --> Garante que o campo de timestamp esteja em formato datetime / Ensures timestamp column is datetime <--

    df_acf = df_acf.sort_values("block_timestamp").set_index("block_timestamp")
    # --> Ordena por timestamp e define como índice / Sorts by timestamp and sets it as index <--

    df_acf = df_acf.resample(resample_freq).mean()
    # --> Reamostra com frequência definida e calcula a média / Resamples at chosen frequency and computes mean <--

    if date_range_start and date_range_end:
        df_acf = df_acf.loc[date_range_start:date_range_end]
    # --> Aplica filtro de datas se fornecido / Applies date range filtering if defined <--

    serie_diff = df_acf["btc_price_usd"].diff().dropna()
    # --> Calcula a série diferenciada de primeira ordem / Computes first-order differencing <--

    if nlags_option == 'manual':
        nlags = manual_nlags
    elif nlags_option == 'serie_longa':
        nlags = len(serie_diff) // 4
    elif nlags_option == 'auto':
        nlags = len(serie_diff)
    else:
        nlags = len(serie_diff)
    # --> Define o número de defasagens para o ACF/PACF com base na escolha / Defines number of lags based on chosen method <--

    # ====================
    # PLOTAGEM ACF e PACF
    # ====================

    acf_vals = acf(serie_diff, nlags=nlags)
    # --> Calcula a ACF da série diferenciada / Computes ACF of the differenced series <--

    pacf_vals = pacf(serie_diff, nlags=nlags)
    # --> Calcula a PACF da série diferenciada / Computes PACF of the differenced series <--

    lags = list(range(1, len(acf_vals)))
    acf_vals_filtered = acf_vals[1:]
    pacf_vals_filtered = pacf_vals[1:]
    # --> Remove o lag 0 para melhor visualização / Removes lag 0 for cleaner visualization <--

    conf = 1.96 / np.sqrt(len(serie_diff))
    # --> Define os limites de confiança (95%) / Sets 95% confidence interval <--

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        subplot_titles=[
            f"ACF da Série Diferenciada — Lag: {nlags_option} ({nlags})",
            f"PACF da Série Diferenciada — Lag: {nlags_option} ({nlags})"
        ]
    )

    # ACF
    fig.add_trace(go.Bar(x=lags, y=acf_vals_filtered, marker_color="#E57C1F", name="ACF"), row=1, col=1)
    fig.add_shape(type="line", x0=1, x1=nlags, y0=conf, y1=conf, line=dict(color="blue", dash="dash"), row=1, col=1)
    fig.add_shape(type="line", x0=1, x1=nlags, y0=-conf, y1=-conf, line=dict(color="blue", dash="dash"), row=1, col=1)
    # --> Adiciona gráfico de barras da ACF com limites de confiança / Adds ACF bars and confidence bounds <--

    # PACF
    fig.add_trace(go.Bar(x=lags, y=pacf_vals_filtered, marker_color="#1FADE5", name="PACF"), row=2, col=1)
    fig.add_shape(type="line", x0=1, x1=nlags, y0=conf, y1=conf, line=dict(color="blue", dash="dash"), row=2, col=1)
    fig.add_shape(type="line", x0=1, x1=nlags, y0=-conf, y1=-conf, line=dict(color="blue", dash="dash"), row=2, col=1)
    # --> Adiciona gráfico de barras da PACF com limites de confiança / Adds PACF bars and confidence bounds <--

    fig.update_layout(
        template="plotly_dark",
        height=800,
        showlegend=False
    )
    # --> Estilização geral do gráfico / General layout styling <--

    fig.update_xaxes(title_text="Defasagem (Lag)", row=2, col=1)
    fig.update_yaxes(title_text="Autocorrelação", row=1, col=1)
    fig.update_yaxes(title_text="Autocorrelação Parcial", row=2, col=1)
    # --> Rótulos dos eixos / Axis labels <--

    fig.show()
    # --> Exibe o gráfico / Displays the plot <--


# 🔶 ₿ ==| plot_arima_layers |============= ₿ ============== | plot_arima_layers |============== ₿ ===============| plot_arima_layers |============ ₿ ============| plot_arima_layers |============ ₿ =============| plot_arima_layers |=====


import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA


def plot_arima_layers(df_acf, p, d, q):
    # =====================
    # PRÉ-PROCESSAMENTO
    # =====================

    serie_original = df_acf["btc_price_usd"].dropna()
    # --> Remove valores nulos da série original / Drops null values from original series <--

    serie_diff = serie_original.diff().dropna()
    # --> Aplica diferenciação de primeira ordem / Applies first-order differencing <--

    modelo = ARIMA(serie_original, order=(p, d, q))
    # --> Inicializa o modelo ARIMA com a série original / Initializes ARIMA model with original series <--

    modelo_ajustado = modelo.fit()
    # --> Ajusta o modelo aos dados / Fits the ARIMA model <--

    fitted_values = modelo_ajustado.fittedvalues
    residuos = modelo_ajustado.resid
    # --> Extrai valores ajustados e resíduos do modelo / Extracts fitted values and residuals <--

    # Ajustar índice para alinhamento
    fitted_values = fitted_values.iloc[1:]
    fitted_values.index = serie_diff.index
    # --> Alinha o índice dos valores ajustados com a série diferenciada / Aligns fitted values index with differenced series <--

    residuos = residuos.iloc[1:]
    residuos.index = serie_diff.index
    # --> Alinha o índice dos resíduos / Aligns residuals index <--

    # ==============
    # PLOTAGEM FINAL
    # ==============
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            "Série Diferenciada vs Ajustada (ARIMA)",
            "Resíduos do Modelo ARIMA",
            "Série Original"
        ]
    )

    fig.add_trace(go.Scatter(
        x=serie_diff.index, y=serie_diff, name="Série Diferenciada",
        line=dict(color="#FFA726", width=1)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=fitted_values.index, y=fitted_values, name="Previsão ARIMA",
        line=dict(color="#26C6DA", width=1)
    ), row=1, col=1)
    # --> Camada 1: compara a série diferenciada real com os valores ajustados / Layer 1: real vs fitted differenced series <--

    fig.add_trace(go.Scatter(
        x=residuos.index, y=residuos, name="Resíduos",
        line=dict(color="#FF7043", width=1)
    ), row=2, col=1)
    # --> Camada 2: visualiza os resíduos do modelo / Layer 2: model residuals <--

    fig.add_trace(go.Scatter(
        x=serie_original.index, y=serie_original, name="Original",
        line=dict(color="#AB47BC", width=1)
    ), row=3, col=1)
    # --> Camada 3: exibe a série original / Layer 3: original series <--

    fig.update_layout(
        height=900,
        title="<b>Visualização das Camadas Extraídas pelo Modelo ARIMA</b>",
        template="plotly_dark",
        showlegend=True
    )

    fig.update_xaxes(title="Data", row=3)
    fig.update_yaxes(title="Valor", row=1)
    fig.update_yaxes(title="Resíduo", row=2)
    fig.update_yaxes(title="Preço BTC (USD)", row=3)

    fig.show()


# 🔶 ₿ ==| plot_acf_residuos |============= ₿ ============== | plot_acf_residuos |============== ₿ ===============| plot_acf_residuos |============ ₿ ============| plot_acf_residuos |============ ₿ =============| plot_acf_residuos |=====


import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf

def plot_acf_residuos(modelo_ajustado, nlags=40):
    """
    Plota o gráfico da ACF dos resíduos de um modelo ARIMA ajustado.

    Parâmetros:
    - modelo_ajustado: modelo ARIMA já ajustado com .fit()
    - nlags: número de defasagens a serem exibidas no gráfico (default=40)
    """

    # ================================
    # EXTRAÇÃO DOS RESÍDUOS
    # ================================

    residuos = modelo_ajustado.resid.dropna()
    # --> Obtém os resíduos do modelo e remove valores nulos / Extracts residuals and drops NaN values <--

    # ================================
    # ACF DOS RESÍDUOS
    # ================================

    acf_vals = acf(residuos, nlags=nlags)
    # --> Calcula a função de autocorrelação até o número de lags definido / Computes ACF up to defined number of lags <--

    conf = 1.96 / np.sqrt(len(residuos))
    # --> Limite de confiança para identificar significância / Confidence interval threshold <--

    lags = list(range(1, len(acf_vals)))
    acf_vals_filtered = acf_vals[1:]
    # --> Remove o lag zero (autocorrelação perfeita) / Removes lag-0 from ACF <--

    fig = go.Figure()
    fig.add_trace(go.Bar(x=lags, y=acf_vals_filtered, marker_color="#E57C1F", name="ACF dos Resíduos"))
    # --> Plota barras da ACF dos resíduos / Plots ACF bars of residuals <--

    fig.add_shape(type="line", x0=1, x1=nlags, y0=conf, y1=conf,
                  line=dict(color="blue", dash="dash"))
    fig.add_shape(type="line", x0=1, x1=nlags, y0=-conf, y1=-conf,
                  line=dict(color="blue", dash="dash"))
    # --> Linhas horizontais: intervalos de confiança de 95% / 95% confidence interval lines <--

    fig.update_layout(
        template="plotly_dark",
        title="ACF dos Resíduos do Modelo ARIMA",
        xaxis_title="Defasagem (Lag)",
        yaxis_title="Autocorrelação",
        height=500
    )

    fig.show()
    # --> Exibe o gráfico interativo / Displays the interactive plot <--


# 🔶 ₿ ==| plot_residuos_analysis |============= ₿ ============== | plot_residuos_analysis |============== ₿ ===============| plot_residuos_analysis |============ ₿ ============| plot_residuos_analysis |============ ₿ =============| plot_residuos_analysis |=====


import numpy as np
import plotly.graph_objects as go
from scipy import stats


def plot_residuos_analysis(residuos):
    """
    Gera dois gráficos:
    1. Histograma com curva normal dos resíduos.
    2. Série temporal dos resíduos.

    Parâmetros:
    - residuos: pandas Series dos resíduos do modelo ARIMA.
    """

    # ==========================
    # DISTRIBUIÇÃO DOS RESÍDUOS
    # ==========================

    media_res = np.mean(residuos)
    # --> Calcula a média dos resíduos / Computes the mean of residuals <--

    desvio_res = np.std(residuos)
    # --> Calcula o desvio padrão dos resíduos / Computes the standard deviation <--

    x_vals = np.linspace(residuos.min(), residuos.max(), 500)
    # --> Gera pontos para o eixo X da curva normal / Generates X values for normal curve <--

    y_vals = stats.norm.pdf(x_vals, media_res, desvio_res)
    # --> Calcula a curva de densidade da distribuição normal teórica / Computes normal PDF curve <--

    fig_hist = go.Figure()

    fig_hist.add_trace(go.Histogram(
    x=residuos,
    nbinsx=50,
    histnorm="probability density",
    name="Resíduos",
    marker=dict(
        color="rgba(229, 165, 0, 0.6)",  # --> Cor translúcida para preenchimento
        line=dict(
            color="#E57C1F",            # --> Cor do contorno (borda da barra)
            width=1.2                   # --> Espessura da borda
         )
        )
    ))

    # --> Histograma dos resíduos normalizado como densidade / Residuals histogram (density normalized) <--

    fig_hist.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        name="Distribuição Normal",
        line=dict(color="white", width=2)
    ))
    # --> Sobrepõe curva de distribuição normal para comparação / Overlays normal distribution curve <--

    fig_hist.update_layout(
        title="Distribuição dos Resíduos vs Normal",
        template="plotly_dark",
        xaxis_title="Valor dos Resíduos",
        yaxis_title="Densidade",
        height=500
    )

    fig_hist.show()

    # ===========================
    # RESÍDUOS AO LONGO DO TEMPO
    # ===========================

    fig_resid = go.Figure()

    fig_resid.add_trace(go.Scatter(
        x=residuos.index,
        y=residuos,
        mode='lines',
        line=dict(color="orange"),
        name="Resíduos"
    ))
    # --> Série temporal dos resíduos (espera-se ruído branco) / Residual time series (expect white noise) <--

    fig_resid.add_shape(
        type="line",
        x0=residuos.index[0],
        x1=residuos.index[-1],
        y0=0,
        y1=0,
        line=dict(color="white", dash="dash")
    )
    # --> Linha de referência no zero / Reference line at y=0 <--

    fig_resid.update_layout(
        title="Resíduos ao Longo do Tempo",
        template="plotly_dark",
        xaxis_title="Data",
        yaxis_title="Valor do Resíduo",
        height=500
    )

    fig_resid.show()


# 🔶 ₿ ==| plot_btc_candlestick_ohlc |============= ₿ ============== | plot_btc_candlestick_ohlc |============== ₿ ===============| plot_btc_candlestick_ohlc |============ ₿ ============| plot_btc_candlestick_ohlc |============ ₿ =============| plot_btc_candlestick_ohlc |=====


import pandas as pd
import plotly.graph_objects as go

def plot_btc_candlestick_ohlc(df_graph, image_path, resample):
    """
    Gera um gráfico de Candlestick com agregação diária (OHLC) do preço do Bitcoin.

    Parâmetros:
    - df_graph (pd.DataFrame): DataFrame contendo coluna 'block_timestamp' e 'btc_price_usd'
    - image_path (str): Caminho da imagem do ícone do Bitcoin
    """

    # ===========================
    # PREPARAÇÃO DOS DADOS
    # ===========================
    
    df_graph = df_graph.resample(resample).mean()

    # Corrigir erro de coluna duplicada no reset_index
    if "block_timestamp" not in df_graph.columns:
        df_graph = df_graph.reset_index(drop=False)
    # --> Garante que 'block_timestamp' esteja como coluna, sem sobrescrever / Ensures it's a column safely <--

    df_graph["block_timestamp"] = pd.to_datetime(df_graph["block_timestamp"])
    # --> Garante que esteja em formato datetime / Ensures datetime format <--
    

    # ===========================
    # CONVERSÃO PARA PADRÃO OHLC
    # ===========================

    df_candles = df_graph.resample("d", on="block_timestamp").agg({
        "btc_price_usd": ["first", "max", "min", "last"]
    }).dropna()
    # --> OHLC por dia (abertura, máxima, mínima e fechamento) / Daily OHLC aggregation <--

    df_candles.columns = ["open", "high", "low", "close"]
    df_candles = df_candles.reset_index()
    # --> Ajusta nomes e traz index para coluna / Rename and flatten structure <--

    # ========================
    # PLOTAR CANDLESTICK
    # ========================
    fig = go.Figure(data=[
        go.Candlestick(
            x=df_candles["block_timestamp"],
            open=df_candles["open"],
            high=df_candles["high"],
            low=df_candles["low"],
            close=df_candles["close"],
            increasing_line_color="#E57C1F",  # --> Laranja para alta / Orange for bullish candles <--
            decreasing_line_color="gray"      # --> Cinza para queda / Gray for bearish candles <--
        )
    ])

    # ==========================
    # INSERIR IMAGEM NO GRÁFICO
    # ==========================
    fig.add_layout_image(
        dict(
            source=image_path,
            xref="paper", yref="paper",
            x=0.008, y=1.1,
            sizex=0.15, sizey=0.15,
            xanchor="left", yanchor="top"
        )
    )
    # --> Adiciona ícone do Bitcoin como marca d’água / Adds Bitcoin icon as visual watermark <--

    # ==================
    # AJUSTE DO LAYOUT
    # ==================
    fig.update_layout(
        title="Gráfico de Candlesticks do Bitcoin",
        xaxis_title="Data",
        yaxis_title="Preço BTC (USD)",
        template="plotly_dark",
        xaxis_rangeslider_visible=False  # --> Remove o slider de tempo abaixo do gráfico / Hide the bottom date range slider <--
    )

    fig.show()
    # --> Exibe o gráfico interativo / Displays the interactive chart <--


# 🔶 ₿ ==| plot_btc_boxplot_by_hour |============= ₿ ============== | plot_btc_boxplot_by_hour |============== ₿ ===============| plot_btc_boxplot_by_hour |============ ₿ ============| plot_btc_boxplot_by_hour |============ ₿ =============| plot_btc_boxplot_by_hour |=====


import plotly.express as px
import pandas as pd

def plot_btc_boxplot_by_hour(df, image_path):
    """
    Gera um boxplot da sazonalidade do preço do Bitcoin por hora do dia.

    Parâmetros:
    - df (pd.DataFrame): DataFrame com colunas ['hour', 'btc_price_usd']
    - image_path (str): Caminho da imagem para marca d'água (ícone do Bitcoin)
    """

    # ========================
    # VERIFICA E EXTRAI A HORA
    # ========================
    if "hour" not in df.columns:
        if "block_timestamp" in df.columns:
            df["hour"] = pd.to_datetime(df["block_timestamp"]).dt.hour
        elif "date" in df.columns:
            df["hour"] = pd.to_datetime(df["date"]).dt.hour
        elif isinstance(df.index, pd.DatetimeIndex):
            df["hour"] = df.index.hour
        else:
            raise ValueError("Não foi possível detectar coluna de timestamp para extrair 'hour'.")
    # --> Extrai a hora do dia automaticamente, mesmo que a coluna não esteja presente <--

    # ===============================================
    # Cálculo dos limites do eixo Y com 2% de buffer
    # ===============================================
    y_min = df["btc_price_usd"].min() * 0.98  # --> 2% abaixo do mínimo / 2% below min <--
    y_max = df["btc_price_usd"].max() * 1.02  # --> 2% acima do máximo / 2% above max <--

    # ===============================
    # BOXPLOT: SAZONALIDADE POR HORA
    # ===============================
    fig = px.box(
        df,
        x="hour",  # --> Eixo X representa a hora do dia / X-axis represents hour of day <--
        y="btc_price_usd",  # --> Eixo Y mostra os valores do Bitcoin / Y-axis shows Bitcoin price values <--
        labels={"hour": "Hora do Dia", "btc_price_usd": "Preço BTC"},
        template="plotly_dark"
    )

    # ================================
    # Estilo das caixas e marcadores
    # ================================
    fig.update_traces(
        line=dict(color="#E57C1F"),   # --> Cor das bordas das caixas / Color of box edges <--
        marker=dict(color="#E57C1F")  # --> Cor dos pontos de outlier / Outlier marker color <--
    )

    # =======================
    # LAYOUT E ESTILO VISUAL
    # =======================
    fig.update_layout(
        height=600,
        showlegend=False,
        title={
            "text": '<b><span style="font-size:22px;">Sazonalidade por Hora do Dia</span></b>',
            "x": 0.5, "y": 0.94,
            "xanchor": "center",
            "yanchor": "top"
        },
        yaxis=dict(
            title="Valor do Bitcoin - USD",
            side="left",
            range=[y_min, y_max]  # --> Ajuste automático com buffer de 2% / Auto range with 2% buffer <--
        ),
        xaxis=dict(
            title="Hora do Dia"  # --> Eixo X com horas / X-axis with hour of day <--
        )
    )

    # ========================
    # INSERIR LOGO DO BITCOIN
    # ========================
    fig.add_layout_image(
        dict(
            source=image_path,
            xref="paper", yref="paper",
            x=0.008, y=1.1,
            sizex=0.1, sizey=0.1,
            xanchor="left", yanchor="top"
        )
    )
    # --> Insere imagem no canto superior esquerdo como marca / Adds BTC image on top-left corner <--

    fig.show()
    # --> Exibe o gráfico / Displays the chart <--


# 🔶 ₿ ==| plot_btc_boxplot_by_month |============= ₿ ============== | plot_btc_boxplot_by_month |============== ₿ ===============| plot_btc_boxplot_by_month |============ ₿ ============| plot_btc_boxplot_by_month |============ ₿ =============| plot_btc_boxplot_by_month |=====


import plotly.express as px
import pandas as pd

def plot_btc_boxplot_by_month(df, image_path):
    """
    Gera um boxplot mensal do preço do Bitcoin, com agrupamento por ano-mês (YYYY-MM).

    Parâmetros:
    - df (pd.DataFrame): DataFrame com colunas ['block_timestamp', 'btc_price_usd']
    - image_path (str): Caminho da imagem a ser usada como marca d’água no gráfico
    """

    # ===========================
    # PREPARAÇÃO DOS DADOS
    # ===========================

    # Corrigir erro de coluna duplicada no reset_index
    if "block_timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df["block_timestamp"] = df.index
        else:
            df = df.reset_index(drop=False)
    # --> Garante que 'block_timestamp' esteja como coluna, sem sobrescrever / Ensures it's a column safely <--

    df["block_timestamp"] = pd.to_datetime(df["block_timestamp"])
    # --> Garante que esteja em formato datetime / Ensures datetime format <--

    # ===============================
    # CRIAR COLUNA "YYYY-MM" MENSAL
    # ===============================
    df["year_month"] = df["block_timestamp"].dt.to_period("M").astype(str)
    # --> Cria uma coluna com o período "YYYY-MM" para análise mensal / Creates a column with "YYYY-MM" period for monthly analysis <--

    df = df.sort_values("year_month")
    # --> Garante que os dados estejam ordenados cronologicamente / Ensures chronological ordering of data <--

    # ===============================
    # LIMITES DO EIXO Y COM BUFFER
    # ===============================
    y_min = df["btc_price_usd"].min() * 0.98
    y_max = df["btc_price_usd"].max() * 1.02
    # --> Define uma margem de 2% abaixo/acima para melhor visualização / Sets a 2% margin below/above for Y-axis <--

    # ================
    # BOXPLOT MENSAL
    # ================
    fig = px.box(
        df,
        x="year_month",
        y="btc_price_usd",
        labels={"year_month": "Ano-Mês", "btc_price_usd": "Preço BTC"},
        template="plotly_dark"
    )
    # --> Cria um boxplot com agrupamento mensal / Creates monthly grouped boxplot <--

    fig.update_traces(
        line=dict(color="#E57C1F"), 
        marker=dict(color="#E57C1F")
    )
    # --> Aplica cor laranja à linha e aos marcadores / Applies orange color to line and markers <--

    # ====================
    # LAYOUT DO GRÁFICO
    # ====================
    fig.update_layout(
        height=600,
        showlegend=False,
        title={
            "text": '<b><span style="font-size:22px;">Sazonalidade do Preço do Bitcoin por Mês</span></b>',
            "x": 0.5, "y": 0.94,
            "xanchor": "center", "yanchor": "top"
        },
        yaxis=dict(
            title="Valor do Bitcoin - USD",
            side="left",
            range=[y_min, y_max]
        ),
        xaxis=dict(
            title="Ano-Mês",
            tickangle=45
        )
    )
    # --> Define layout geral do gráfico: altura, títulos e eixo X inclinado / Sets layout: height, titles, angled X-axis <--

    # ========================
    # INSERIR IMAGEM NO TOPO
    # ========================
    fig.add_layout_image(
        dict(
            source=image_path,
            xref="paper", yref="paper",
            x=0.008, y=1.1,
            sizex=0.1, sizey=0.1,
            xanchor="left", yanchor="top"
        )
    )
    # --> Insere logotipo do Bitcoin como marca visual / Adds Bitcoin logo as decorative watermark <--

    fig.show()
    # --> Exibe o gráfico interativo / Displays the interactive chart <--


# 🔶 ₿ ==| plot_log_return_analysis |============= ₿ ============== | plot_log_return_analysis |============== ₿ ===============| plot_log_return_analysis |============ ₿ ============| plot_log_return_analysis |============ ₿ =============| plot_log_return_analysis |=====


import numpy as np
import plotly.graph_objects as go
from scipy import stats

def plot_log_return_analysis(log_return):
    """
    Gera dois gráficos interativos para análise de retornos logarítmicos:
    1. Histograma com curva de densidade normal.
    2. Série temporal dos retornos logarítmicos.

    Parâmetros:
    - log_returns: pandas Series com retornos logarítmicos.
    """

    # ===========================
    # HISTOGRAMA + CURVA NORMAL
    # ===========================

    media = np.mean(log_return)
    # --> Média dos retornos / Mean of log returns <--

    desvio = np.std(log_return)
    # --> Desvio padrão dos retornos / Standard deviation of log returns <--

    x_vals = np.linspace(log_return.min(), log_return.max(), 500)
    # --> Eixo X para a curva normal / X-axis values for normal curve <--

    y_vals = stats.norm.pdf(x_vals, media, desvio)
    # --> Densidade teórica normal / Normal PDF <--

    fig_hist = go.Figure()

    fig_hist.add_trace(go.Histogram(
        x=log_return,
        nbinsx=50,
        histnorm="probability density",
        name="Resíduos",
        marker=dict(
        color="rgba(229, 165, 0, 0.6)",  # --> Cor translúcida para preenchimento
        line=dict(
            color="#E57C1F",            # --> Cor do contorno (borda da barra)
            width=1.2                   # --> Espessura da borda
         )
        )
    ))

    # --> Histograma dos retornos logarítmicos / Histogram of log returns <--

    fig_hist.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        name="Distribuição Normal",
        line=dict(color="white", width=2)
    ))
    # --> Sobreposição da curva normal / Overlay normal curve <--

    fig_hist.update_layout(
        title="Distribuição dos Retornos Logarítmicos vs Normal",
        template="plotly_dark",
        xaxis_title="Retorno Logarítmico",
        yaxis_title="Densidade",
        height=500
    )

    fig_hist.show()

    # ==============================
    # SÉRIE TEMPORAL DOS RETORNOS
    # ==============================

    fig_return = go.Figure()

    fig_return.add_trace(go.Scatter(
        x=log_return.index,
        y=log_return,
        mode='lines',
        line=dict(color="#E57C1F"),
        name="Log Retorno"
    ))
    # --> Série temporal dos retornos logarítmicos / Time series of log returns <--

    fig_return.add_shape(
        type="line",
        x0=log_return.index[0],
        x1=log_return.index[-1],
        y0=0,
        y1=0,
        line=dict(color="white", dash="dash")
    )
    # --> Linha de referência em y=0 / Reference line at y=0 <--

    fig_return.update_layout(
        title="Retornos Logarítmicos ao Longo do Tempo",
        template="plotly_dark",
        xaxis_title="Data",
        yaxis_title="Log Retorno",
        height=500
    )

    fig_return.show()


# 🔶 ₿ ==| plot_acf_pacf_returns |============= ₿ ============== | plot_acf_pacf_returns |============== ₿ ===============| plot_acf_pacf_returns |============ ₿ ============| plot_acf_pacf_returns |============ ₿ =============| plot_acf_pacf_returns |=====


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf, pacf
from plotly.subplots import make_subplots

# ================================================================
# PLOTAGEM DE ACF E PACF PARA RETORNOS LOGARÍTMICOS
# ACF & PACF PLOTTING FOR LOG RETURNS
# ================================================================
def plot_acf_pacf_returns(series: pd.Series, nlags: int = 30):
    """
    Gera gráficos interativos de ACF e PACF para uma série de retornos.
    / Generates interactive ACF and PACF plots for a return series.

    Parâmetros / Parameters:
    - series: pd.Series contendo os retornos logarítmicos / Series with log returns.
    - nlags: número máximo de defasagens a serem plotadas / Max number of lags to plot.
    """

    # ==========================
    # PREPARAÇÃO DOS DADOS / DATA CLEANING
    # ==========================
    series = series.dropna()
    # --> Remove valores ausentes / Drop missing values <--

    if len(series) < 3:
        print(f"[AVISO] Série muito curta (n = {len(series)}) para ACF/PACF. Requer pelo menos 3 pontos.")
        return
        # --> Série precisa de no mínimo 3 pontos para autocorrelação / Minimum 3 points required <--

    nlags = min(nlags, len(series) - 1)
    # --> Ajusta número de lags conforme tamanho da série / Adjusts lag count to series length <--

    try:
        acf_vals = acf(series, nlags=nlags)
        pacf_vals = pacf(series, nlags=nlags)
        # --> Calcula ACF e PACF / Compute ACF and PACF <--

        lags = list(range(1, nlags + 1))
        acf_vals_filtered = acf_vals[1:]
        pacf_vals_filtered = pacf_vals[1:]
        # --> Remove lag 0 para melhor visualização / Remove lag 0 for clearer plots <--

        conf = 1.96 / np.sqrt(len(series))
        # --> Intervalo de confiança 95% / 95% confidence interval <--

        # ==========================
        # PLOTAGEM COM PLOTLY / INTERACTIVE PLOTTING
        # ==========================
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=False,
            subplot_titles=["ACF - Retornos Logarítmicos", "PACF - Retornos Logarítmicos"]
        )

        # ACF
        fig.add_trace(go.Bar(x=lags, y=acf_vals_filtered, marker_color="#E57C1F", name="ACF"), row=1, col=1)
        fig.add_shape(type="line", x0=1, x1=nlags, y0=conf, y1=conf, line=dict(color="blue", dash="dash"), row=1, col=1)
        fig.add_shape(type="line", x0=1, x1=nlags, y0=-conf, y1=-conf, line=dict(color="blue", dash="dash"), row=1, col=1)

        # PACF
        fig.add_trace(go.Bar(x=lags, y=pacf_vals_filtered, marker_color="#1FADE5", name="PACF"), row=2, col=1)
        fig.add_shape(type="line", x0=1, x1=nlags, y0=conf, y1=conf, line=dict(color="blue", dash="dash"), row=2, col=1)
        fig.add_shape(type="line", x0=1, x1=nlags, y0=-conf, y1=-conf, line=dict(color="blue", dash="dash"), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=800,
            showlegend=False
        )

        fig.update_xaxes(title_text="Defasagem (Lag)", row=2, col=1)
        fig.update_yaxes(title_text="Autocorrelação", row=1, col=1)
        fig.update_yaxes(title_text="Autocorrelação Parcial", row=2, col=1)

        fig.show()

    except Exception as e:
        print(f"[ERRO] Falha ao gerar ACF/PACF: {e}")
        # --> Erro durante o cálculo ou plotagem / Catch failure in calc or plot <--


# 🔶 ₿ ==| plot_volatility_rolling |============= ₿ ============== | plot_volatility_rolling |============== ₿ ===============| plot_volatility_rolling |============ ₿ ============| plot_volatility_rolling |============ ₿ =============| plot_volatility_rolling |=====


import pandas as pd
import plotly.graph_objects as go
import numpy as np

# ================================================================
# PLOTAGEM DE VOLATILIDADE ROLLING / ROLLING VOLATILITY PLOT
# ================================================================
def plot_volatility_rolling(df, window=24):
    """
    Gera um gráfico interativo da volatilidade (rolling std) dos retornos logarítmicos.
    / Generates an interactive plot of volatility (rolling std) of log returns.

    Parâmetros / Parameters:
    - df: pd.DataFrame contendo a coluna 'log_return' / DataFrame with 'log_return' column.
    - window: int, janela de tempo usada no cálculo da volatilidade / Rolling window size.
    """

    df = df.copy()

    # ================================
    # VALIDAÇÃO DE ENTRADA / INPUT VALIDATION
    # ================================
    if "log_return" not in df.columns:
        print("[ERRO] A coluna 'log_return' não foi encontrada no DataFrame.")
        return
        # --> Verifica se a coluna esperada existe / Checks if required column exists <--

    if df["log_return"].dropna().shape[0] < window:
        print(f"[AVISO] Série insuficiente para calcular volatilidade com janela {window}.")
        return
        # --> Série muito curta para aplicar o rolling / Too short for rolling computation <--

    # ================================
    # CÁLCULO DA VOLATILIDADE / VOLATILITY COMPUTATION
    # ================================
    df["volatility_rolling"] = df["log_return"].rolling(window).std()
    df = df.dropna(subset=["volatility_rolling"])
    # --> Aplica média móvel e remove NaNs / Applies rolling std and drops NaNs <--

    # ================================
    # CRIAÇÃO DO GRÁFICO / FIGURE CREATION
    # ================================
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["volatility_rolling"],
        mode="lines",
        line=dict(color="#E57C1F", width=1.5),
        fill='tozeroy',
        fillcolor="rgba(229, 165, 0, 0.1)",
        name="Volatilidade"
    ))
    # --> Linha com preenchimento suave para a volatilidade / Smooth area fill <--

    fig.update_layout(
        title="<b><span style='font-size:20px;'>Volatilidade Rolling dos Retornos Logarítmicos</span></b>",
        template="plotly_dark",
        height=500,
        showlegend=False,
        xaxis_title="Data",
        yaxis_title="Volatilidade (Desvio Padrão Rolling)"
    )
    # --> Estilo escuro e rótulos organizados / Dark theme and labeled axes <--

    fig.show()  


# 🔶 ₿ ==| plot_stl_decomposition |============= ₿ ============== | plot_stl_decomposition |============== ₿ ===============| plot_stl_decomposition |============ ₿ ============| plot_stl_decomposition |============ ₿ =============| plot_stl_decomposition |=====


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# ================================================================
# VISUALIZAÇÃO DA DECOMPOSIÇÃO STL / STL DECOMPOSITION PLOT
# ================================================================
def plot_stl_decomposition(
    df_stl,
    price_series: pd.Series = None,
    price_col: str = "btc_price_usd",
    trend_col: str = "trend",
    seasonal_col: str = "seasonal",
    resid_col: str = "resid",
    image_path: str = None
):
    """
    Gera um gráfico interativo com os componentes da decomposição STL.
    / Generates an interactive chart with STL decomposition components.
    """

    # ================================
    # INSERE A SÉRIE ORIGINAL SE FOR FORNECIDA
    # / ADD ORIGINAL PRICE SERIES IF PROVIDED
    # ================================
    if price_series is not None:
        df_stl = df_stl.copy()
        
        if len(price_series) != len(df_stl):
            raise ValueError("A série de preços e o DataFrame STL devem ter o mesmo comprimento.")
        
        df_stl[price_col] = price_series.reset_index(drop=True).iloc[:len(df_stl)]
        # --> Força alinhamento posicional com reset_index / Forces positional alignment with reset_index <--

    # ================================
    # VALIDA COLUNAS REQUERIDAS
    # ================================
    required_cols = [price_col, trend_col, seasonal_col, resid_col]
    for col in required_cols:
        if col not in df_stl.columns:
            raise ValueError(f"A coluna '{col}' não está presente no DataFrame STL.")

    # ================================
    # CONFIGURAÇÃO DO GRÁFICO / PLOT SETUP
    # ================================
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Série Original", "Tendência", "Sazonalidade", "Resíduo")
    )

    y_min = df_stl[price_col].min() * 0.98
    y_max = df_stl[price_col].max() * 1.02

    # ================================
    # COMPONENTES STL / STL COMPONENTS
    # ================================
    components = [
        (price_col, 1, "Original", "rgba(229, 165, 0, 0.1)", 1.5),
        (trend_col, 2, "Tendência", None, 1.5),
        (seasonal_col, 3, "Sazonalidade", None, 1.5),
        (resid_col, 4, "Resíduo", None, 1.2),
    ]

    for col, row, name, fill, width in components:
        fig.add_trace(go.Scatter(
            x=df_stl.index,
            y=df_stl[col],
            mode="lines",
            name=name,
            line=dict(color="#E57C1F", width=width),
            fill="tozeroy" if fill else None,
            fillcolor=fill
        ), row=row, col=1)

    # ================================
    # LAYOUT FINAL / FINAL LAYOUT
    # ================================
    fig.update_layout(
        template="plotly_dark",
        height=900,
        showlegend=False,
        title=dict(
            text="<b><span style='font-size:22px;'>Decomposição STL do Preço do Bitcoin</span></b>",
            x=0.5,
            y=0.97,
            xanchor="center"
        ),
        yaxis=dict(title="Preço BTC (USD)", range=[y_min, y_max])
    )

    fig.update_yaxes(title_text="Preço BTC (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Tendência", row=2, col=1)
    fig.update_yaxes(title_text="Sazonal", row=3, col=1)
    fig.update_yaxes(title_text="Resíduo", row=4, col=1)
    fig.update_xaxes(title_text="Data", row=4, col=1)

    # ================================
    # MARCA D'ÁGUA / WATERMARK
    # ================================
    if image_path:
        fig.add_layout_image(dict(
            source=image_path,
            xref="paper", yref="paper",
            x=0.008, y=1.12,
            sizex=0.1, sizey=0.1,
            xanchor="left", yanchor="top"
        ))

    fig.show()


# 🔶 ₿ ==| plot_fft_spectrum |============= ₿ ============== | plot_fft_spectrum |============== ₿ ===============| plot_fft_spectrum |============ ₿ ============| plot_fft_spectrum |============ ₿ =============| plot_fft_spectrum |=====


import plotly.graph_objects as go
import numpy as np

def plot_fft_spectrum(
    periods: np.ndarray,
    power: np.ndarray,
    fft_vals: np.ndarray,
    n: int,
    top_peaks_idx: np.ndarray,
    image_path: str = None
):
    """
    Gera gráfico interativo com os principais harmônicos detectados via Transformada de Fourier (FFT).
    / Generates interactive plot showing dominant harmonics detected via Fast Fourier Transform (FFT).
    
    Parâmetros / Parameters:
    - periods (np.ndarray): Vetor de períodos correspondentes às frequências positivas / Periods corresponding to FFT frequencies
    - power (np.ndarray): Potência espectral associada a cada período / Spectral power for each period
    - fft_vals (np.ndarray): Coeficientes complexos da FFT (frequência positiva) / FFT complex coefficients (positive frequencies only)
    - n (int): Número de pontos na série original / Number of original time series points
    - top_peaks_idx (np.ndarray): Índices das frequências dominantes / Indices of top dominant peaks
    - image_path (str): Caminho para o logo/imagem a ser inserido no gráfico (opcional) / Path to image/logo to embed (optional)
    """

    # ================================
    # VISUALIZAÇÃO DOS HARMÔNICOS FFT
    # ================================

    fig_fft = go.Figure()

    # --- Linha preenchida com degradê âmbar translúcido / Amber shaded area line ---
    fig_fft.add_trace(go.Scatter(
        x=periods,
        y=power,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(229, 165, 0, 0.25)",
        line=dict(color="#E57C1F", width=2),
        name="Potência / Power"
    ))

    # --- Anotações dos picos mais relevantes / Annotate top dominant peaks ---
    for idx in top_peaks_idx:
        amplitude = np.abs(fft_vals[idx]) / n
        periodo = periods[idx]
        potencia = power[idx]
        horas = round(periodo * 0.5, 1)
        dias = round(horas / 24, 1)

        tooltip = (
            f"<b>{horas}h ({dias}d)</b><br>"
            f"Ciclo recorrente a cada {dias} dias<br>"
            f"Amplitude média: ±{amplitude:.2f} USD<br>"
            f"Potência: {round(potencia, 2)} bilhões USD²"
        )

        fig_fft.add_trace(go.Scatter(
            x=[periodo],
            y=[potencia],
            mode="markers+text",
            marker=dict(color="white", size=8),
            text=[f"{horas}h"],
            textposition="top center",
            textfont=dict(size=12, color="white"),
            hovertemplate=tooltip,
            showlegend=False
        ))

    # --- Estilização geral do gráfico / Overall styling and layout ---
    fig_fft.update_layout(
        title={
            "text": "<b><span style='font-size:22px;'>Harmônicos (Ciclos) Dominantes no Preço do Bitcoin</span><br><span style='font-size:14px;'>Detectados via Transformada de Fourier (FFT)</span></b>",
            "x": 0.5, "y": 0.95,
            "xanchor": "center", "yanchor": "top"
        },
        xaxis_title="Período (em passos de 30 min) / Period (in 30-min steps)",
        yaxis_title="Intensidade do Harmônico (em USD²) / Harmonic Intensity (in USD²)",
        template="plotly_dark",
        height=600,
        margin=dict(l=60, r=40, t=100, b=50)
    )

    fig_fft.update_yaxes(tickformat=",", title_font=dict(size=14), tickfont=dict(size=12))
    fig_fft.update_xaxes(title_font=dict(size=14), tickfont=dict(size=12))

    # --- Marca d’água (opcional) / Optional watermark ---
    if image_path:
        fig_fft.add_layout_image(
            dict(
                source=image_path,
                xref="paper", yref="paper",
                x=0.008, y=1.2,
                sizex=0.15, sizey=0.15,
                xanchor="left", yanchor="top"
            )
        )

    fig_fft.show()
    # --> Exibe o gráfico interativo final / Displays the final interactive FFT plot <--


# 🔶 ₿ ==| plot_seasonal_daily_line |============= ₿ ============== | plot_seasonal_daily_line |============== ₿ ===============| plot_seasonal_daily_line |============ ₿ ============| plot_seasonal_daily_line |============ ₿ =============| plot_seasonal_daily_line |=====


import pandas as pd
import plotly.express as px

def plot_seasonal_daily_line(
    df: pd.DataFrame,
    timestamp_col: str = "block_timestamp",
    price_col: str = "btc_price_usd",
    ano_alvo: int = 2024
):
    """
    Gera gráfico de linha sazonal diário com a média de preços do Bitcoin por dia do mês.
    / Plots a seasonal daily line chart with average Bitcoin prices by day of month.

    Parâmetros / Parameters:
    - df (pd.DataFrame): DataFrame com timestamp e coluna de preço / DataFrame with timestamp and price column
    - timestamp_col (str): Nome da coluna de data/hora / Timestamp column name
    - price_col (str): Nome da coluna de preços / Price column name
    - ano_alvo (int): Ano para filtrar e exibir no gráfico / Target year to filter and plot
    """

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    # --> Garante que o campo de tempo esteja no formato datetime / Ensures timestamp is in datetime format <--

    # ====================
    # ENRIQUECIMENTO TEMPORAL
    # ====================

    df["year"] = df[timestamp_col].dt.year
    df["month"] = df[timestamp_col].dt.month
    df["day"] = df[timestamp_col].dt.day
    # --> Extrai ano, mês e dia da data / Extracts year, month and day from timestamp <--

    meses = {
        1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr",
        5: "Mai", 6: "Jun", 7: "Jul", 8: "Ago",
        9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
    }
    df["month_name"] = df["month"].map(meses)
    df["month_name"] = pd.Categorical(df["month_name"], categories=list(meses.values()), ordered=True)
    # --> Mapeia número do mês para nome e ordena categoricamente / Maps month number to name and orders <--

    # ====================
    # AGRUPAMENTO DIÁRIO
    # ====================
    df_diario = df.groupby(["year", "month_name", "day"], observed=False)[price_col].mean().reset_index()
    # --> Média do preço por dia do mês para cada mês/ano / Daily average price per month/year <--

    df_filtrado = df_diario[df_diario["year"] == ano_alvo]
    # --> Filtra para o ano desejado / Filters for selected year <--

    # ====================
    # PLOTAGEM DO GRÁFICO
    # ====================

    fig = px.line(
        df_filtrado,
        x="day",
        y=price_col,
        color="month_name",
        labels={
            "day": "Dia do Mês",
            price_col: "Preço Médio (USD)",
            "month_name": "Mês"
        },
        title="Gráfico Sazonal Diário: Preço Médio do Bitcoin por Dia do Mês"
    )

    fig.update_layout(
        template="plotly_dark",
        height=600,
        xaxis_title="Dia do Mês",
        yaxis_title="Preço Médio (USD)",
        legend_title="Mês"
    )
    # --> Aplica estilo escuro com legendas personalizadas / Applies dark style with custom axis and legend <--

    fig.show()
    # --> Exibe o gráfico final / Displays the final chart <--


# 🔶 ₿ ==| plot_seasonal_daily_line |============= ₿ ============== | plot_seasonal_daily_line |============== ₿ ===============| plot_seasonal_daily_line |============ ₿ ============| plot_seasonal_daily_line |============ ₿ =============| plot_seasonal_daily_line |=====


import pandas as pd
import plotly.express as px

def plot_seasonal_weekly_line(
    df: pd.DataFrame,
    timestamp_col: str = "block_timestamp",
    price_col: str = "btc_price_usd",
    ano_alvo: int = 2025,
    image_path: str = None
):
    """
    Gera gráfico de linha sazonal semanal com a média de preços do Bitcoin por dia da semana.
    / Plots a seasonal weekly line chart with average Bitcoin prices by weekday across weeks.

    Parâmetros / Parameters:
    - df (pd.DataFrame): DataFrame com timestamp e coluna de preço / DataFrame with timestamp and price column
    - timestamp_col (str): Nome da coluna de data/hora / Timestamp column name
    - price_col (str): Nome da coluna de preços / Price column name
    - ano_alvo (int): Ano para filtrar os dados / Target year to filter
    - image_path (str): Caminho para a imagem de marca d’água (opcional) / Path to watermark image (optional)
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    # --> Converte o campo de tempo para datetime / Converts timestamp to datetime <--

    # ==============================
    # EXTRAÇÃO DE ATRIBUTOS TEMPORAIS
    # ==============================

    df["week"] = df[timestamp_col].dt.isocalendar().week
    df["year"] = df[timestamp_col].dt.year
    df["weekday"] = df[timestamp_col].dt.dayofweek
    # --> Extrai semana, ano e dia da semana / Extracts ISO week, year, and day of week <--

    dias_semana = {
        0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui",
        4: "Sex", 5: "Sáb", 6: "Dom"
    }
    df["weekday_name"] = df["weekday"].map(dias_semana)
    df["weekday_name"] = pd.Categorical(df["weekday_name"], categories=list(dias_semana.values()), ordered=True)
    # --> Mapeia nomes dos dias da semana e define ordem categórica / Maps and orders weekday names <--

    # =============================
    # AGRUPAMENTO POR SEMANA E DIA
    # =============================

    df_grouped = df.groupby(["year", "week", "weekday_name"], observed=False)[price_col].mean().reset_index()
    # --> Agrupa por ano, semana e dia da semana / Groups by year, week, and weekday <--

    df_filtered = df_grouped[df_grouped["year"] == ano_alvo]
    # --> Filtra apenas o ano desejado / Filters for selected year <--

    # =============================
    # CRIAÇÃO DO GRÁFICO INTERATIVO
    # =============================

    fig = px.line(
        df_filtered,
        x="weekday_name",
        y=price_col,
        color="week",
        labels={
            "weekday_name": "Dia da Semana",
            price_col: "Preço Médio (USD)",
            "week": "Semana"
        }
    )

    # ================================
    # ESTILIZAÇÃO DO LAYOUT DO GRÁFICO
    # ================================

    fig.update_layout(
        template="plotly_dark",
        height=800,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        showlegend=True,
        bargap=0.05,
        title={
            "text": "<b><span style='font-size:22px;'>Gráfico Sazonal Semanal: Preço Médio do Bitcoin por Dia da Semana</span></b>",
            "x": 0.5, "y": 0.95,
            "xanchor": "center", "yanchor": "top"
        },
        xaxis_title="Dias da Semana",
        yaxis_title="Preço do Bitcoin - USD",
        legend_title="Semana do Ano",
        yaxis=dict(title="Preço do Bitcoin - USD", side="left"),
    )

    # ====================
    # MARCA D'ÁGUA (Opcional)
    # ====================
    if image_path:
        fig.add_layout_image(
            dict(
                source=image_path,
                xref="paper", yref="paper",
                x=0.008, y=1.06,
                sizex=0.1, sizey=0.1,
                xanchor="left", yanchor="top"
            )
        )
        # --> Adiciona imagem do Bitcoin no canto superior esquerdo / Adds Bitcoin watermark to top left <--

    fig.show()
    # --> Exibe o gráfico final / Displays the final plot <--


# 🔶 ₿ ==| plot_weekly_seasonality_all_years |============= ₿ ============== | plot_weekly_seasonality_all_years |============== ₿ ===============| plot_weekly_seasonality_all_years |============ ₿ ============| plot_weekly_seasonality_all_years |============ ₿ =============| plot_weekly_seasonality_all_years |=====


import pandas as pd
import plotly.express as px

def plot_weekly_seasonality_all_years(
    df: pd.DataFrame,
    timestamp_col: str = "block_timestamp",
    price_col: str = "btc_price_usd",
    facet_col_wrap: int = 2
):
    """
    Gera gráfico de sazonalidade semanal por dia da semana para todos os anos presentes no DataFrame.
    / Plots weekly seasonality of Bitcoin prices by weekday, separated by year.

    Parâmetros / Parameters:
    - df (pd.DataFrame): DataFrame com colunas de timestamp e preço / DataFrame with timestamp and price
    - timestamp_col (str): Nome da coluna de timestamp / Name of timestamp column
    - price_col (str): Nome da coluna de preços / Name of price column
    - facet_col_wrap (int): Número de colunas por linha no facet / Number of columns per row in facet layout
    """

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    # --> Garante formato datetime para o timestamp / Ensures datetime format for timestamp <--

    # ========================
    # EXTRAÇÃO DE ATRIBUTOS TEMPORAIS
    # ========================
    df["week"] = df[timestamp_col].dt.isocalendar().week
    df["year"] = df[timestamp_col].dt.year
    df["weekday"] = df[timestamp_col].dt.dayofweek
    # --> Extrai semana ISO, ano e dia da semana / Extracts ISO week, year, weekday <--

    dias_semana = {
        0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui",
        4: "Sex", 5: "Sáb", 6: "Dom"
    }
    df["weekday_name"] = df["weekday"].map(dias_semana)
    df["weekday_name"] = pd.Categorical(df["weekday_name"], categories=list(dias_semana.values()), ordered=True)
    # --> Mapeia nomes dos dias da semana em ordem / Maps and orders weekday names <--

    # ========================
    # AGRUPAMENTO POR ANO, SEMANA E DIA
    # ========================
    df_grouped = df.groupby(["year", "week", "weekday_name"], observed=False)[price_col].mean().reset_index()
    # --> Agrupa por ano, semana e nome do dia da semana / Groups by year, week, and weekday name <--

    # ========================
    # PLOTAGEM COM FACETAGEM POR ANO
    # ========================
    fig = px.line(
        df_grouped,
        x="weekday_name",
        y=price_col,
        color="week",
        line_group="week",  # --> Garante que linhas de semanas não se conectem entre anos / Ensures weekly lines stay independent by year <--
        facet_col="year",   # --> Cria subgráficos por ano / Facets plots by year <--
        facet_col_wrap=facet_col_wrap,
        labels={
            "weekday_name": "Dia da Semana",
            price_col: "Preço Médio (USD)",
            "week": "Semana",
            "year": "Ano"
        },
        title="Gráfico Sazonal Semanal: Preço Médio do Bitcoin por Dia da Semana (Todos os Anos)"
    )

    # ========================
    # ESTILIZAÇÃO DO GRÁFICO
    # ========================
    fig.update_layout(
        template="plotly_dark",
        height=700,
        showlegend=True,
        xaxis_title="Dia da Semana",
        yaxis_title="Preço Médio (USD)",
        legend_title="Semana do Ano"
    )
    # --> Estilo escuro, títulos dos eixos e legenda / Dark theme, axes and legend styling <--

    fig.show()
    # --> Exibe o gráfico final interativo / Displays the final interactive chart <--


# 🔶 ₿ ==| plot_intraday_price_by_hour |============= ₿ ============== | plot_intraday_price_by_hour |============== ₿ ===============| plot_intraday_price_by_hour |============ ₿ ============| plot_intraday_price_by_hour |============ ₿ =============| plot_intraday_price_by_hour |=====


import pandas as pd
import plotly.express as px

def plot_intraday_price_by_hour(
    df: pd.DataFrame,
    year_filter: int,
    timestamp_col: str = "block_timestamp",
    price_col: str = "btc_price_usd",
    image_path: str = None
):
    """
    Gera gráfico intradiário do preço médio do Bitcoin por hora, com uma linha por dia.
    / Generates intraday line plot of average Bitcoin price per hour, one line per day.

    Parâmetros / Parameters:
    - df (pd.DataFrame): DataFrame com colunas de timestamp e preço / DataFrame with timestamp and price columns
    - year_filter (int): Ano para filtrar os dados (ex: 2025) / Year to filter (e.g., 2025)
    - timestamp_col (str): Nome da coluna de tempo / Name of the timestamp column
    - price_col (str): Nome da coluna de preço / Name of the price column
    - image_path (str): Caminho para a imagem de marca d'água (opcional) / Path to watermark image (optional)
    """

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    # --> Garante formato datetime / Ensures datetime format <--

    # ==========================
    # COLUNAS AUXILIARES TEMPORAIS
    # ==========================
    df["year"] = df[timestamp_col].dt.year
    df["date"] = df[timestamp_col].dt.date
    df["hour"] = df[timestamp_col].dt.hour
    # --> Extrai ano, data e hora / Extracts year, date, and hour <--

    # ==========================
    # AGRUPAMENTO POR DIA E HORA
    # ==========================
    df_grouped = df.groupby(["year", "date", "hour"], observed=False)[price_col].mean().reset_index()
    # --> Média do preço por hora para cada dia / Computes hourly average price per day <--

    df_filtered = df_grouped[df_grouped["year"] == year_filter]
    # --> Filtra apenas o ano desejado / Filters the selected year <--

    # ==========================
    # PLOTAGEM COM PLOTLY
    # ==========================
    fig = px.line(
        df_filtered,
        x="hour",
        y=price_col,
        color="date",
        labels={
            "hour": "Hora do Dia / Hour of Day",
            price_col: "Preço Médio (USD) / Avg. Price (USD)",
            "date": "Data / Date"
        },
        title=f"Análise Intra-Diária: Preço Médio por Hora ({year_filter})"
    )

    fig.update_layout(
        template="plotly_dark",
        height=800,
        xaxis=dict(dtick=1),
        yaxis_title="Preço do Bitcoin - USD / Bitcoin Price - USD",
        xaxis_title="Hora do Dia / Hour of Day",
        legend_title="Data",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        title={
            "text": f"<b><span style='font-size:22px;'>Análise Intra-Diária: Preço Médio por Hora ({year_filter})</span></b>",
            "x": 0.5, "y": 0.95,
            "xanchor": "center", "yanchor": "top"
        }
    )
    # --> Layout visual escuro e responsivo / Clean dark responsive layout <--

    # ==========================
    # INSERÇÃO DE IMAGEM (OPCIONAL)
    # ==========================
    if image_path:
        fig.add_layout_image(
            dict(
                source=image_path,
                xref="paper", yref="paper",
                x=0.008, y=1.06,
                sizex=0.08, sizey=0.08,
                xanchor="left", yanchor="top"
            )
        )
        # --> Adiciona logotipo visual como marca d’água / Adds logo as visual watermark <--

    fig.show()
    # --> Exibe o gráfico interativo / Displays the interactive plot <--


# 🔶 ₿ ==| plot_bitcoin_seasonal_patterns |============= ₿ ============== | plot_bitcoin_seasonal_patterns |============== ₿ ===============| plot_bitcoin_seasonal_patterns |============ ₿ ============| plot_bitcoin_seasonal_patterns |============ ₿ =============| plot_bitcoin_seasonal_patterns |=====


import pandas as pd
import plotly.express as px


def plot_bitcoin_seasonal_patterns(df: pd.DataFrame) -> None:
    """
    Gera três gráficos interativos com foco na sazonalidade trimestral e mensal do preço do Bitcoin.
    / Generates three interactive plots focused on the quarterly and monthly seasonality of Bitcoin prices.

    Parâmetros / Parameters:
    - df: DataFrame com colunas ['block_timestamp', 'btc_price_usd']
           contendo dados históricos de preços do Bitcoin.
           / DataFrame with ['block_timestamp', 'btc_price_usd'] columns.
    """

    # ======================================
    # PRÉ-PROCESSAMENTO DOS CAMPOS TEMPORAIS
    # ======================================
    df = df.copy()
    df["block_timestamp"] = pd.to_datetime(df["block_timestamp"])
    df["year"] = df["block_timestamp"].dt.year
    df["quarter"] = df["block_timestamp"].dt.quarter
    df["month"] = df["block_timestamp"].dt.month
    df["month_name"] = df["block_timestamp"].dt.strftime("%b")
    df["month_name"] = pd.Categorical(
        df["month_name"],
        categories=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        ordered=True
    )
    df["quarter_start"] = df["block_timestamp"].dt.to_period("Q").dt.start_time

    # ================================
    # 1. PREÇO MÉDIO POR TRIMESTRE
    # ================================
    df_quarter = df.groupby("quarter_start")["btc_price_usd"].mean().reset_index()
    fig1 = px.line(
        df_quarter,
        x="quarter_start", y="btc_price_usd",
        title="<b>Preço Médio do Bitcoin por Trimestre</b>",
        labels={"quarter_start": "Trimestre", "btc_price_usd": "Preço Médio (USD)"},
        template="plotly_dark"
    )
    fig1.show()

    # =====================================================
    # 2. PADRÕES SAZONAIS: TRIMESTRES AGRUPADOS POR ANO
    # =====================================================
    df_season = df.groupby(["year", "quarter"], observed=False)["btc_price_usd"].mean().reset_index()
    fig2 = px.line(
        df_season,
        x="quarter", y="btc_price_usd", color="year",
        labels={
            "quarter": "Trimestre",
            "btc_price_usd": "Preço Médio (USD)",
            "year": "Ano"
        },
        title="<b>Padrões Sazonais Trimestrais do Bitcoin</b>",
        template="plotly_dark"
    )
    fig2.show()

    # =============================================================
    # 3. SUBSÉRIES MENSAIS: EVOLUÇÃO DO PREÇO POR MÊS (FACETADAS)
    # =============================================================
    df_sub = df.groupby(["month_name", "year"], observed=False)["btc_price_usd"].mean().reset_index()
    fig3 = px.line(
        df_sub,
        x="year", y="btc_price_usd", facet_col="month_name",
        facet_col_wrap=4,
        labels={
            "year": "Ano",
            "btc_price_usd": "Preço Médio (USD)",
            "month_name": "Mês"
        },
        title="<b>Gráfico de Subséries: Evolução do Preço do Bitcoin por Mês</b>",
        template="plotly_dark"
    )
    fig3.update_layout(height=800)
    fig3.show()




# 🔶 ₿ ==| plot_btc_boxplot_by_month_comparison |============= ₿ ============== | plot_btc_boxplot_by_month_comparison |============== ₿ ===============| plot_btc_boxplot_by_month_comparison |============ ₿ ============| plot_btc_boxplot_by_month_comparison |============ ₿ =============| plot_btc_boxplot_by_month_comparison |=====


import plotly.express as px

def plot_btc_boxplot_by_month_comparison(df, image_path):
    """
    Gera um boxplot do preço do Bitcoin por mês do ano, comparando entre 2024 e 2025 com cores distintas.

    Parâmetros:
    - df (pd.DataFrame): DataFrame com colunas ['block_timestamp', 'btc_price_usd']
    - image_path (str): Caminho da imagem a ser usada como marca d’água no gráfico
    """

    # ===========================
    # PREPARAÇÃO DOS DADOS
    # ===========================
    if "block_timestamp" not in df.columns:
        df = df.reset_index(drop=False)

    df["block_timestamp"] = pd.to_datetime(df["block_timestamp"])
    df["Ano"] = df["block_timestamp"].dt.year
    df["Mês"] = df["block_timestamp"].dt.month_name()

    # Ordenar meses
    meses_ordem = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    df["Mês"] = pd.Categorical(df["Mês"], categories=meses_ordem, ordered=True)

    # ===========================
    # LIMITES DO EIXO Y COM BUFFER
    # ===========================
    y_min = df["btc_price_usd"].min() * 0.98
    y_max = df["btc_price_usd"].max() * 1.02

    # ===========================
    # BOXPLOT MENSAL
    # ===========================
    fig = px.box(
        df,
        x="Mês",
        y="btc_price_usd",
        color="Ano",
        labels={"Mês": "Mês", "btc_price_usd": "Preço BTC (USD)", "Ano": "Ano"},
        template="plotly_dark",
        category_orders={"Mês": meses_ordem},
        color_discrete_map={
            2024: "#1F77B4",  # azul para 2024
            2025: "#E57C1F"   # laranja para 2025
        }
    )

    # ===========================
    # LAYOUT
    # ===========================
    fig.update_layout(
        height=600,
        boxmode="group",
        title={
            "text": '<b><span style="font-size:22px;">Boxplot: Preço do Bitcoin por Mês (Comparado por Ano)</span></b>',
            "x": 0.5, "y": 0.94,
            "xanchor": "center", "yanchor": "top"
        },
        yaxis=dict(
            title="Preço do Bitcoin - USD",
            range=[y_min, y_max]
        ),
        xaxis=dict(
            title="Mês",
            tickangle=45
        )
    )

    # ===========================
    # INSERIR IMAGEM
    # ===========================
    fig.add_layout_image(
        dict(
            source=image_path,
            xref="paper", yref="paper",
            x=0.008, y=1.1,
            sizex=0.1, sizey=0.1,
            xanchor="left", yanchor="top"
        )
    )

    fig.show()


# 🔶 ₿ ==| plot_btc_boxplot_by_week_comparison |============= ₿ ============== | plot_btc_boxplot_by_week_comparison |============== ₿ ===============| plot_btc_boxplot_by_week_comparison |============ ₿ ============| plot_btc_boxplot_by_week_comparison |============ ₿ =============| plot_btc_boxplot_by_week_comparison |=====


import plotly.express as px

def plot_btc_boxplot_by_week_comparison(df, image_path):
    """
    Gera um boxplot do preço do Bitcoin por semana do ano, comparando os anos de forma visual.

    Parâmetros:
    - df (pd.DataFrame): DataFrame com colunas ['block_timestamp', 'btc_price_usd']
    - image_path (str): Caminho da imagem a ser usada como marca d’água no gráfico
    """

    # ===========================
    # PREPARAÇÃO DOS DADOS
    # ===========================
    if "block_timestamp" not in df.columns:
        df = df.reset_index(drop=False)

    df["block_timestamp"] = pd.to_datetime(df["block_timestamp"])
    df["Ano"] = df["block_timestamp"].dt.year
    df["Semana"] = df["block_timestamp"].dt.isocalendar().week

    # ===========================
    # LIMITES DO EIXO Y COM BUFFER
    # ===========================
    y_min = df["btc_price_usd"].min() * 0.98
    y_max = df["btc_price_usd"].max() * 1.02

    # ===========================
    # BOXPLOT SEMANAL
    # ===========================
    fig = px.box(
        df,
        x="Semana",
        y="btc_price_usd",
        color="Ano",
        labels={"Semana": "Semana do Ano", "btc_price_usd": "Preço BTC (USD)", "Ano": "Ano"},
        template="plotly_dark",
        color_discrete_map={
            2024: "#1F77B4",  # azul
            2025: "#E57C1F"   # laranja
        }
    )

    # ===========================
    # LAYOUT
    # ===========================
    fig.update_layout(
        height=600,
        boxmode="group",
        title={
            "text": '<b><span style="font-size:22px;">Boxplot: Preço do Bitcoin por Semana do Ano (Comparado por Ano)</span></b>',
            "x": 0.5, "y": 0.94,
            "xanchor": "center", "yanchor": "top"
        },
        yaxis=dict(
            title="Preço do Bitcoin - USD",
            range=[y_min, y_max]
        ),
        xaxis=dict(
            title="Semana do Ano",
            tickmode="linear",
            tick0=1,
            dtick=1
        )
    )

    # ===========================
    # INSERIR IMAGEM
    # ===========================
    fig.add_layout_image(
        dict(
            source=image_path,
            xref="paper", yref="paper",
            x=0.008, y=1.1,
            sizex=0.1, sizey=0.1,
            xanchor="left", yanchor="top"
        )
    )

    fig.show()


# 🔶 ₿ ==| plot_rolling_diagnostics_overlay |============= ₿ ============== | plot_rolling_diagnostics_overlay |============== ₿ ===============| plot_rolling_diagnostics_overlay |============ ₿ ============| plot_rolling_diagnostics_overlay |============ ₿ =============| plot_rolling_diagnostics_overlay |=====


import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_rolling_diagnostics_overlay(
    series: pd.Series,
    window: int = 30,
    title: str = "Rolling Média e Desvio (Overlay)",
    image_path: str = None
):
    """
    Gera gráfico interativo com a série original, média móvel e desvio padrão sobrepostos.
    / Generates interactive Plotly chart with overlaid original, rolling mean and std.
    """

    mean_roll = series.rolling(window).mean()
    std_roll = series.rolling(window).std()

    fig = make_subplots(
        rows=1, cols=1,
        shared_xaxes=True,
        subplot_titles=("Rolling Média e Desvio - Série Sobreposta",)
    )

    # Série Original
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series,
        mode="lines",
        name="Original",
        line=dict(color="#E57C1F", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(229, 165, 0, 0.05)"
    ))

    # Média Móvel
    fig.add_trace(go.Scatter(
        x=mean_roll.index,
        y=mean_roll,
        mode="lines",
        name="Média Móvel",
        line=dict(color="cyan", width=2, dash="dot")
    ))

    # Desvio Padrão Móvel
    fig.add_trace(go.Scatter(
        x=std_roll.index,
        y=std_roll,
        mode="lines",
        name="Desvio Padrão Móvel",
        line=dict(color="magenta", width=2, dash="dash")
    ))

    # Layout Final com legenda à direita
    fig.update_layout(
        template="plotly_dark",
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v",
            y=1,
            yanchor="top",
            x=1.02,
            xanchor="left"
        ),
        title={
            "text": f'<b><span style="font-size:22px;">{title}</span></b>',
            "x": 0.5,
            "y": 0.93,
            "xanchor": "center"
        }
    )

    fig.update_yaxes(title_text="Valor")
    fig.update_xaxes(title_text="Data")

    # Marca d'água (opcional)
    if image_path:
        fig.add_layout_image(
            dict(
                source=image_path,
                xref="paper", yref="paper",
                x=0.008, y=1.12,
                sizex=0.1, sizey=0.1,
                xanchor="left", yanchor="top"
            )
        )

    fig.show()


# 🔶 ₿ ==| plot_rolling_diagnostics_overlay |============= ₿ ============== | plot_rolling_diagnostics_overlay |============== ₿ ===============| plot_rolling_diagnostics_overlay |============ ₿ ============| plot_rolling_diagnostics_overlay |============ ₿ =============| plot_rolling_diagnostics_overlay |=====


import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_transformed_series(result_dict: dict, title: str = "Transformações da Série Temporal"):
    """
    Gera gráfico interativo com a série original, 1ª e 2ª diferença.
    / Generates an interactive Plotly chart showing original, 1st diff and 2nd diff.

    Parâmetros:
    - result_dict: dicionário retornado por apply_series_transformations()
    - title: título do gráfico
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Série Original", "1ª Diferença", "2ª Diferença")
    )

    # =======================
    # SÉRIE ORIGINAL
    # =======================
    fig.add_trace(go.Scatter(
        x=result_dict["original"].index,
        y=result_dict["original"].values,
        mode="lines",
        name="Original",
        line=dict(color="#E57C1F", width=1.5),
        fillcolor="rgba(229, 165, 0, 0.1)"
    ), row=1, col=1)

    # =======================
    # 1ª DIFERENÇA
    # =======================
    fig.add_trace(go.Scatter(
        x=result_dict["diff_1st"].index,
        y=result_dict["diff_1st"].values,
        mode="lines",
        name="1ª Diferença",
        line=dict(color="#E57C1F", width=1.5)
    ), row=2, col=1)

    # =======================
    # 2ª DIFERENÇA
    # =======================
    fig.add_trace(go.Scatter(
        x=result_dict["diff_2nd"].index,
        y=result_dict["diff_2nd"].values,
        mode="lines",
        name="2ª Diferença",
        line=dict(color="#E57C1F", width=1.5)
    ), row=3, col=1)

    # =======================
    # LAYOUT FINAL
    # =======================
    fig.update_layout(
        template="plotly_dark",
        height=800,
        showlegend=False,
        title={
            "text": f'<b><span style="font-size:22px;">{title}</span></b>',
            "x": 0.5,
            "xanchor": "center"
        }
    )

    fig.update_yaxes(title_text="Original", row=1, col=1)
    fig.update_yaxes(title_text="1ª Diferença", row=2, col=1)
    fig.update_yaxes(title_text="2ª Diferença", row=3, col=1)
    fig.update_xaxes(title_text="Data", row=3, col=1)

    fig.show()


# 🔶 ₿ ==| plot_transaction_fee_time_series |============= ₿ ============== | plot_transaction_fee_time_series |============== ₿ ===============| plot_transaction_fee_time_series |============ ₿ ============| plot_transaction_fee_time_series |============ ₿ =============| plot_transaction_fee_time_series |=====


import plotly.express as px
import pandas as pd

def plot_transaction_fee_time_series(df_transactions, image_path):
    """
    Gera um gráfico de linha com a série temporal das taxas médias de transação por hora,
    com uma linha suavizada e marca d'água do ícone do BTC.

    Parâmetros:
    - df_transactions (pd.DataFrame): DataFrame com as transações e as taxas ('fee').
    - image_path (str): Caminho para o arquivo de imagem a ser usado como marca d'água.
    """
    # Filtra as transações e calcula a média de taxas por hora
    df_transactions_filtered = df_transactions.select(
        F.date_format("timestamp", "yyyy-MM-dd HH").alias("hour"), 
        "fee"
    )
    # --> Seleciona o timestamp e a taxa, agrupando por hora / Selects the timestamp and fee, grouping by hour <--

    df_transactions_hourly = df_transactions_filtered.groupBy("hour") \
        .agg(F.avg("fee").alias("avg_fee")) \
        .orderBy("hour")
    # --> Calcula a média da taxa por hora / Calculates the average fee per hour <--

    # Convertendo para Pandas e suavizando a média das taxas com uma janela de 6 horas
    df_transactions_hourly_pd = df_transactions_hourly.toPandas()
    df_transactions_hourly_pd["smoothed_fee"] = df_transactions_hourly_pd["avg_fee"].rolling(window=6, min_periods=1).mean()
    # --> Suaviza as taxas médias utilizando uma média móvel / Smooths the average fees using a rolling mean <--

    # Criando o gráfico de linha
    fig = px.line(
        df_transactions_hourly_pd, 
        x="hour", 
        y="avg_fee",  
        title="Evolução das Taxas Médias de Transação por Hora",
        labels={"hour": "Data", "avg_fee": "Taxa Média (BTC)"},
        template="plotly_dark"
    )
    # --> Cria o gráfico de linha com a taxa média / Creates the line chart with average fee <--

    fig.update_traces(
        line=dict(color="#E57C1F", width=1.5),
        fill='tozeroy',
        fillcolor="rgba(229, 165, 0, 0.3)"
    )
    # --> Atualiza a linha do gráfico para exibir a taxa média com preenchimento / Updates the chart line to display the average fee with fill <--

    # Adicionando a marca d'água da imagem
    fig.add_layout_image(
        dict(
            source=image_path,
            xref="paper", yref="paper",
            x=0.008, y=1.1,  
            sizex=0.15, sizey=0.15,
            xanchor="left", yanchor="top"
        )
    )
    # --> Adiciona ícone do Bitcoin como marca d'água / Adds Bitcoin logo as visual watermark <--

    fig.update_layout(
        height=600,
        showlegend=False,
        title={
            "text": '<b><span style="font-size:22px;">Média das Taxas de Transação ao Longo do Tempo (₿ Média Móvel)</span></b>',
            "x": 0.5, "y": 0.92,  
            "xanchor": "center", "yanchor": "top"
        },
        yaxis=dict(
            title="Média da Taxa (BTC)", 
            side="right"
        ),
        xaxis=dict(
            title="Data"
        )
    )
    # --> Atualiza o layout do gráfico, incluindo título e eixo Y / Updates the chart layout including title and Y-axis <--

    fig.show()
    # --> Exibe o gráfico / Displays the chart <--


# 🔶 ₿ ==| plot_btc_moved_hourly |============= ₿ ============== | plot_btc_moved_hourly |============== ₿ ===============| plot_btc_moved_hourly |============ ₿ ============| plot_btc_moved_hourly |============ ₿ =============| plot_btc_moved_hourly |=====


import plotly.express as px
import pandas as pd

def plot_btc_total_hourly(df_blocks, image_path):
    """
    Gera um gráfico de linha com a soma do valor total de Bitcoin por hora, 
    suavizado com uma média móvel de 6 horas, e com marca d'água do ícone do BTC.

    Parâmetros:
    - df_blocks (pd.DataFrame): DataFrame contendo os dados de blocos com coluna 'timestamp' e 'block_value_btc'.
    - image_path (str): Caminho para o arquivo de imagem a ser usado como marca d'água.
    """
    # Filtra os blocos por hora e soma o valor de Bitcoin por hora
    df_blocks_filtered = df_blocks.select(F.date_format("timestamp", "yyyy-MM-dd HH").alias("hour"), "block_value_btc")
    # --> Agrupa os blocos por hora e soma o valor de Bitcoin / Groups blocks by hour and sums Bitcoin value <--

    df_blocks_hourly = df_blocks_filtered.groupBy("hour") \
        .agg(F.sum("block_value_btc").alias("block_value_btc")) \
        .orderBy("hour")
    # --> Calcula a soma do valor total de Bitcoin por hora / Calculates the total Bitcoin value per hour <--

    # Convertendo para Pandas e suavizando os valores com uma média móvel de 6 horas
    df_blocks_hourly_pd = df_blocks_hourly.toPandas()
    df_blocks_hourly_pd["smoothed_btc"] = df_blocks_hourly_pd["block_value_btc"].rolling(window=6, min_periods=1).mean()
    # --> Aplica a média móvel para suavizar os valores / Applies rolling mean to smooth the values <--

    # Criando o gráfico de linha
    fig = px.line(
        df_blocks_hourly_pd, 
        x="hour", 
        y="smoothed_btc", 
        title="Bitcoin BTC - Valor Total por Hora (₿ Média Móvel)",
        labels={"hour": "Data", "smoothed_btc": "BTC"},
        template="plotly_dark"
    )
    # --> Cria o gráfico de linha para visualizar a soma suavizada do valor de Bitcoin por hora / Creates the line chart to visualize the smoothed total Bitcoin value per hour <--

    fig.update_traces(
        line=dict(color="#E57C1F", width=1.5),
        fill='tozeroy',
        fillcolor="rgba(229, 165, 0, 0.3)"
    )
    # --> Atualiza o gráfico com cor e preenchimento / Updates the chart with color and fill <--

    # Adicionando a marca d'água da imagem
    fig.add_layout_image(
        dict(
            source=image_path,
            xref="paper", yref="paper",
            x=0.008, y=1.1,  
            sizex=0.15, sizey=0.15,
            xanchor="left", yanchor="top"
        )
    )
    # --> Adiciona ícone do Bitcoin como marca d'água / Adds Bitcoin logo as visual watermark <--

    # Atualiza o layout do gráfico
    fig.update_layout(
        height=600,
        showlegend=False,
        title={
            "text": '<b><span style="font-size:22px;">Bitcoin BTC - Valor Total por Hora (₿ Média Móvel)</span></b>',
            "x": 0.5, "y": 0.92,  
            "xanchor": "center", "yanchor": "top"
        },
        yaxis=dict(
            title="BTC Total(₿)", 
            side="right"
        )
    )
    # --> Atualiza o título e o eixo Y do gráfico / Updates the title and Y-axis of the chart <--

    fig.show()
    # --> Exibe o gráfico / Displays the chart <--


# 🔶 ₿ ==| plot_btc_moved_hourly |============= ₿ ============== | plot_transaction_fee_time_series |============== ₿ ===============| plot_transaction_fee_time_series |============ ₿ ============| plot_transaction_fee_time_series |============ ₿ =============| plot_transaction_fee_time_series |=====


import plotly.express as px
import pandas as pd

def plot_btc_moved_hourly(df_blocks, image_path):
    """
    Gera um gráfico de linha com o valor total de Bitcoin movimentado por hora, 
    suavizado com uma média móvel de 6 horas, e com marca d'água do ícone do BTC.

    Parâmetros:
    - df_blocks (pd.DataFrame): DataFrame com os dados dos blocos, incluindo 'timestamp' e 'total_btc_moved'.
    - image_path (str): Caminho para o arquivo de imagem a ser usado como marca d'água.
    """
    # Filtra os blocos e calcula a soma do valor total movimentado de Bitcoin por hora
    df_blocks_filtered = df_blocks.select(F.date_format("timestamp", "yyyy-MM-dd HH").alias("hour"), "total_btc_moved")
    # --> Agrupa os blocos por hora e soma o valor total movimentado de Bitcoin / Groups blocks by hour and sums the total Bitcoin moved <--

    df_blocks_hourly = df_blocks_filtered.groupBy("hour") \
        .agg(F.sum("total_btc_moved").alias("total_btc_moved")) \
        .orderBy("hour")
    # --> Calcula o valor total de Bitcoin movimentado por hora / Calculates the total Bitcoin moved per hour <--

    # Convertendo para Pandas e suavizando os valores com uma média móvel de 6 horas
    df_blocks_hourly_pd = df_blocks_hourly.toPandas()
    df_blocks_hourly_pd["smoothed_btc"] = df_blocks_hourly_pd["total_btc_moved"].rolling(window=6, min_periods=1).mean()
    # --> Aplica a média móvel para suavizar os valores de BTC movimentados / Applies rolling mean to smooth the Bitcoin moved values <--

    # Criando o gráfico de linha
    fig = px.line(
        df_blocks_hourly_pd, 
        x="hour", 
        y="smoothed_btc", 
        title="Bitcoin BTC - Valor Total Movimentado por Hora (₿ Média Móvel)",
        labels={"hour": "Data", "smoothed_btc": "BTC Movimentado"},
        template="plotly_dark"
    )
    # --> Cria o gráfico de linha para visualizar a soma suavizada do valor de Bitcoin movimentado por hora / Creates the line chart to visualize the smoothed total Bitcoin moved per hour <--

    fig.update_traces(
        line=dict(color="#E57C1F", width=1.5),
        fill='tozeroy',
        fillcolor="rgba(229, 165, 0, 0.3)"
    )
    # --> Atualiza o gráfico com cor e preenchimento / Updates the chart with color and fill <--

    # Adicionando a marca d'água da imagem
    fig.add_layout_image(
        dict(
            source=image_path,
            xref="paper", yref="paper",
            x=0.008, y=1.1,  
            sizex=0.15, sizey=0.15,
            xanchor="left", yanchor="top"
        )
    )
    # --> Adiciona ícone do Bitcoin como marca d'água / Adds Bitcoin logo as visual watermark <--

    # Atualiza o layout do gráfico
    fig.update_layout(
        height=600,
        showlegend=False,
        title={
            "text": '<b><span style="font-size:22px;">Bitcoin BTC - Valor Total Movimentado por Hora (₿ Média Móvel)</span></b>',
            "x": 0.5, "y": 0.92,  
            "xanchor": "center", "yanchor": "top"
        },
        yaxis=dict(
            title="BTC Movimentado (₿)", 
            side="right"
        )
    )
    # --> Atualiza o título e o eixo Y do gráfico / Updates the title and Y-axis of the chart <--

    fig.show()
    # --> Exibe o gráfico / Displays the chart <--


# 🔶 ₿ ==| plot_btc_blocks_per_day |============= ₿ ============== | plot_btc_blocks_per_day |============== ₿ ===============| plot_btc_blocks_per_day |============ ₿ ============| plot_btc_blocks_per_day |============ ₿ =============| plot_btc_blocks_per_day |=====


import plotly.express as px
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def plot_btc_blocks_per_day(df_blocks, image_path):
    """
    Gera um gráfico com a frequência de blocos minerados por dia usando dados do DataFrame Spark e Plotly.

    Parâmetros:
    - df_blocks (pyspark.sql.DataFrame): DataFrame com colunas 'block_height' e 'timestamp'.
    - image_path (str): Caminho para a imagem a ser usada como marca d'água no gráfico.
    """
    
    # ===============================
    # PRÉ-PROCESSAMENTO DOS DADOS
    # ===============================

    df_blocks_filtered = df_blocks.select(F.col("block_height"), F.col("timestamp"))
    # --> Seleciona apenas as colunas necessárias para a análise / Selects only required columns for analysis <--

    df_blocks_filtered = df_blocks_filtered.withColumn("timestamp", F.col("timestamp").cast("timestamp"))
    # --> Converte a coluna timestamp para tipo Timestamp no Spark / Converts column to Spark TimestampType <--

    df_blocks_daily = df_blocks_filtered \
        .groupBy(F.date_format(F.col("timestamp"), "yyyy-MM-dd").alias("date")) \
        .agg(F.count("block_height").alias("block_count")) \
        .orderBy("date")
    # --> Agrupa os blocos por data e conta quantos foram minerados por dia / Groups blocks by date and counts how many were mined each day <--

    df_blocks_daily_pd = df_blocks_daily.toPandas()
    # --> Converte o DataFrame Spark para Pandas para visualização com Plotly / Converts Spark DataFrame to Pandas for Plotly visualization <--

    # ===============================
    # CONSTRUÇÃO DO GRÁFICO
    # ===============================

    fig = px.line(
        df_blocks_daily_pd,
        x="date",
        y="block_count",
        title="Frequência de Blocos Minerados por Dia (Média Móvel)",
        labels={"date": "Data", "block_count": "Número de Blocos"},
        template="plotly_dark"
    )
    # --> Cria gráfico de linha com a contagem diária de blocos / Creates line chart with daily block counts <--

    fig.update_traces(
        line=dict(color="#E57C1F", width=1.5),
        fill='tozeroy',
        fillcolor="rgba(229, 165, 0, 0.3)"
    )
    # --> Atualiza estilo da linha e adiciona preenchimento abaixo dela / Updates line style and adds area fill <--

    # ===============================
    # INSERIR IMAGEM NO GRÁFICO
    # ===============================

    fig.add_layout_image(
        dict(
            source=image_path,
            xref="paper", yref="paper",
            x=0.008, y=1.1,
            sizex=0.15, sizey=0.15,
            xanchor="left", yanchor="top"
        )
    )
    # --> Adiciona imagem do Bitcoin como marca d’água / Adds Bitcoin logo as watermark <--

    fig.update_layout(
        height=600,
        showlegend=False,
        title={
            "text": '<b><span style="font-size:22px;">Frequência de Blocos Minerados por Dia</span></b>',
            "x": 0.5, "y": 0.92,
            "xanchor": "center", "yanchor": "top"
        },
        yaxis=dict(
            title="Número de Blocos",
            side="right"
        )
    )
    # --> Define layout final do gráfico, incluindo título, eixo Y e altura / Finalizes chart layout: title, Y-axis and height <--

    fig.show()
    # --> Exibe o gráfico / Display the chart <--


# 🔶 ₿ ==| plot_transaction_distribution |============= ₿ ============== | plot_transaction_distribution |============== ₿ ===============| plot_transaction_distribution |============ ₿ ============| plot_transaction_distribution |============ ₿ =============| plot_transaction_distribution |=====


import plotly.express as px
import pandas as pd

def plot_transaction_distribution(df_blocks, image_path):
    """
    Gera um histograma da distribuição do número de transações por bloco,
    com personalização de cores e uma marca d'água do ícone do BTC.

    Parâmetros:
    - df_blocks (pd.DataFrame): DataFrame contendo os dados dos blocos com a coluna 'num_transactions'.
    - image_path (str): Caminho para o arquivo de imagem a ser usado como marca d'água.
    """
    # Convertendo os dados para Pandas para o histograma
    transactions_data = df_blocks.select(F.col("num_transactions")).toPandas()
    # --> Converte os dados do número de transações para Pandas / Converts transaction data to Pandas for histogram <--

    # Criando o histograma
    fig = px.histogram(
        transactions_data,
        x="num_transactions",
        nbins=50,  # Define o número de bins / Sets the number of bins <--
        title="Distribuição do Número de Transações por Bloco",
        labels={"num_transactions": "Número de Transações por Bloco"},
        opacity=0.85
    )
    # --> Cria o histograma da distribuição de transações por bloco / Creates the histogram of transactions per block distribution <--

    fig.update_traces(
        marker=dict(color="rgba(229, 165, 0, 0.7)", line=dict(color="#E57C1F", width=1))
    )
    # --> Atualiza o marcador do histograma com cores personalizadas / Updates histogram marker with custom colors <--

    # Configura o layout do gráfico
    fig.update_layout(
        height=800,
        template="plotly_dark",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        title={
            "text": '<b><span style="font-size:22px;">Bitcoin BTC - Distribuição do Número de Transações por Bloco</span></b>',
            "x": 0.5, "y": 0.92,
            "xanchor": "center", "yanchor": "top"
        },
        bargap=0.05, 
        xaxis_title="Número de Transações",
        yaxis_title="Frequência",
        yaxis=dict(title="Frequência", side="left"),
    )
    # --> Personaliza o layout com fundo escuro, título e rótulos / Customizes the layout with dark background, title, and labels <--

    # Adicionando a marca d'água da imagem
    fig.add_layout_image(
        dict(
            source=image_path,
            xref="paper", yref="paper",
            x=0.008, y=1.1,  
            sizex=0.15, sizey=0.15,
            xanchor="left", yanchor="top"
        )
    )
    # --> Adiciona ícone do Bitcoin como marca d'água / Adds Bitcoin logo as visual watermark <--

    fig.show()
    # --> Exibe o gráfico / Displays the chart <--




import pandas as pd
import plotly.express as px

def plot_btc_boxplot_by_dayofyear(df, image_path):
    """
    Gera boxplot do Preço do Bitcoin por Dia do Ano (MM-DD), ignorando o ano.

    Parâmetros:
    - df: DataFrame com colunas ['block_timestamp', 'btc_price_usd']
    - image_path: Caminho da imagem a ser usada como marca d’água (ícone do Bitcoin)
    """

    # ===========================
    # PREPARAÇÃO DOS DADOS
    # ===========================
    if "block_timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df["block_timestamp"] = df.index
        else:
            df = df.reset_index(drop=False)
    # --> Garante que 'block_timestamp' esteja presente como coluna / Ensures 'block_timestamp' is a column <--

    df["block_timestamp"] = pd.to_datetime(df["block_timestamp"])
    # --> Converte para datetime se necessário / Converts to datetime if needed <--

    # ===============================
    # CRIA A COLUNA "MM-DD"
    # ===============================
    df["month_day"] = pd.Categorical(
        df["block_timestamp"].dt.strftime("%m-%d"),
        ordered=True
    )
    # --> Cria coluna MM-DD e força ordenação correta / Creates MM-DD column as ordered categorical <--

    df = df.sort_values("month_day")
    # --> Ordena os dados cronologicamente por MM-DD / Sorts data by MM-DD order <--

    # ===============================
    # LIMITES DO EIXO Y COM BUFFER
    # ===============================
    y_min = df["btc_price_usd"].min() * 0.98  # --> 2% abaixo do mínimo / 2% below min <--
    y_max = df["btc_price_usd"].max() * 1.02  # --> 2% acima do máximo / 2% above max <--

    # ===============================
    # BOXPLOT POR DIA DO ANO
    # ===============================
    fig = px.box(
        df,
        x="month_day",  # --> Eixo X com MM-DD categórico / X-axis as MM-DD categorical <--
        y="btc_price_usd",
        labels={"month_day": "Dia do Ano (MM-DD)", "btc_price_usd": "Preço BTC"},
        template="plotly_dark"
    )
    # --> Cria gráfico boxplot agrupado por dia do ano / Creates day-of-year grouped boxplot <--

    fig.update_traces(
        line=dict(color="#E57C1F"),
        marker=dict(color="#E57C1F")
    )
    # --> Estiliza a linha e marcadores com cor laranja / Styles boxes with orange color <--

    # ========================
    # LAYOUT DO GRÁFICO
    # ========================
    fig.update_layout(
        height=600,
        showlegend=False,
        title={
            "text": '<b><span style="font-size:22px;">Sazonalidade por Dia do Ano</span></b>',
            "x": 0.5, "y": 0.94,
            "xanchor": "center", "yanchor": "top"
        },
        yaxis=dict(
            title="Valor do Bitcoin - USD",
            side="left",
            range=[y_min, y_max]
        ),
        xaxis=dict(
            title="Dia do Ano (MM-DD)",
            tickangle=45,
            showticklabels=False  # --> Evita poluição visual no eixo X / Avoids visual clutter on X-axis <--
        )
    )
    # --> Define layout visual do gráfico / Sets overall visual layout <--

    # ============================
    # INSERE MARCA D'ÁGUA (BTC)
    # ============================
    fig.add_layout_image(
        dict(
            source=image_path,
            xref="paper", yref="paper",
            x=0.008, y=1.1,
            sizex=0.1, sizey=0.1,
            xanchor="left", yanchor="top"
        )
    )
    # --> Insere o logotipo do Bitcoin como marca d’água / Adds Bitcoin logo as watermark <--

    fig.show()
    # --> Exibe o gráfico / Displays the interactive chart <--

plot_volatility_rolling