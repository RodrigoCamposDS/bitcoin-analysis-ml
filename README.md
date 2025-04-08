# IA Preditiva para o Bitcoin

## Organização dos Dados  

bitcoin_features/  
├── blockchain_blocks_part/  
├── blockchain_transactions_part/  
├── blockchain_addresses_part/  
├── features_temp/  
├── features_dolar_parquet/  
└── block_range_checkpoint.txt 

## Estrutura do Projeto  

project/  
│  
├── notebooks/  
│   ├── init.py
│   ├── 01-blockchain_request_data.ipynb  
│   ├── 02-data-exploration.ipynb  
│   ├── 03-feature-engineering.ipynb  
│   ├── 04-bitcoin-forecasting.ipynb  
│   ├── 05-NLP.ipynb  
│   └── 06-LSTM-model.ipynb  
│
├── src/   
│   ├── crypto_btc.egg-info/  
│   ├── data/
│   │   ├── data_processed/  
│   │   │   └── data_exploration.py  
│   │   ├── etl/  
│   │   │   └── etl_blockchain.py  
│   │   ├── features/  
│   │   │   ├── arima_features.py  
│   │   │   ├── fft_features.py  
│   │   │   ├── regime_features.py  
│   │   │   └── stl_features.py
│   │   └── visualizations/ 
│   │       └── plot_btc.py
│  
├── crypto_btc/    
│   └── init.py
│  
└── README.md   

## Sumário

- [Objetivo do Projeto](#objetivo-do-projeto)
- [Etapa 1: Coleta e Pré-Processamento Assíncrono](#etapa-1-coleta-e-pré-processamento-assíncrono)
- [Etapa 2: Processamento e Salvamento com Spark](#etapa-2-processamento-e-salvamento-com-spark)
- [Etapa 3: Engenharia de Features e Enriquecimento](#etapa-3-engenharia-de-features-e-enriquecimento)
- [Etapa 4: Análise Temporal e Extração de Padrões para Modelagem](#etapa-4-análise-temporal-e-extração-de-padrões-para-modelagem)
- [Organização dos Dados](#organização-dos-dados)
- [Conclusão](#conclusão)
- [Em Andamento](#em-andamento)
- [Próximos Passos](#próximos-passos)

---

## Objetivo do Projeto

O objetivo principal deste projeto é desenvolver uma Inteligência Artificial preditiva, capaz de aprender padrões complexos do mercado de Bitcoin com base em dados históricos, contínuos e ruidosos, utilizando técnicas modernas de Machine Learning e Estatística Avançada.

Este projeto foi idealizado como uma jornada prática e técnica para construir uma pipeline robusta de ETL com foco em dados reais, brutos e contínuos: os dados da blockchain do Bitcoin.

O objetivo central foi aplicar e aprofundar habilidades em:

- Coleta assíncrona de grandes volumes de dados com asyncio e aiohttp
- Extração e modelagem de variáveis úteis ao entendimento do mercado de Bitcoin
- Armazenamento eficiente e particionado com Parquet
- Preparação para ingestão em data warehouses escaláveis como o BigQuery
- Estruturação de pipelines resilientes com controle incremental e modularidade

---

## Etapa 1: Coleta e Pré-Processamento Assíncrono

- Programação assíncrona com asyncio e aiohttp
- Divisão inteligente em lotes (tamanho_lote=100) para simular ingestão contínua
- Suporte a múltiplos modos de coleta: auto, recente, antigo, manual
- Cálculo financeiro por bloco: total_btc_moved, total_fees, block_reward, block_value_btc
- Extração de transações enriquecidas: tx_hash, timestamp, total_input, total_output, fee, transaction_size, num_inputs, num_outputs
- Engenharia de carteiras: inferência heurística, direção do fluxo, flags is_zero, balance_before, balance_after

Pastas geradas:

blockchain_blocks_part/  
blockchain_transactions_part/  
blockchain_addresses_part/  

---

## Etapa 2: Processamento e Salvamento com Spark

- Spark local configurado para simulação com múltiplos núcleos
- Controle incremental via metadata.txt
- Escrita particionada por year e month:
  - processed_data/: histórico acumulado
  - bitcoin_features/: versão atual
- Uso de StorageLevel.MEMORY_AND_DISK

### Validação com PyArrow e Multiprocessamento

- Verificação de esquemas Parquet (pyarrow)
- Correção de arquivos inválidos
- ProcessPoolExecutor para verificação paralela
- Recuperação com asyncio e aiohttp

---

## Etapa 3: Engenharia de Features e Enriquecimento

- Enriquecimento dos blocos com preço do Bitcoin (por timestamp)
- Agregações por transações:
  - avg_fee_per_tx, num_transactions, avg_tx_size, avg_input_per_tx, avg_output_per_tx
- Agregações por endereços:
  - total_input_btc_addr, total_output_btc_addr, unique_addresses, zero_balance_addresses, multisig_wallets

### Técnicas Aplicadas

- Broadcast joins e particionamento
- Checkpoints (block_range_checkpoint.txt)
- Escrita condicional apenas para partições novas

---

## Etapa 4: Análise Temporal e Extração de Padrões para Modelagem

Consolidação de variáveis para modelagem preditiva robusta, com base em teoria de séries temporais e finanças quantitativas.  
Nem todas as variáveis extraídas serão utilizadas diretamente na modelagem inicial. Contudo, todas estão organizadas e documentadas para possibilitar análises futuras, testes de importância, seleção automatizada (XGBoost, LGBM) e refinamento iterativo do pipeline de previsão.

### 1. Setup Inicial
- Inicialização da sessão Spark
- Leitura das partições processadas com base nos checkpoints

### 2. Auditoria e Validação
- Continuidade e granularidade temporal
- Checagem de nulos e consistência de `block_timestamp`
- Inferência automática de `block_height` nos casos de falha

### 3. Análise Exploratória
- Visualização de preços, retornos e volatilidade
- Análise de densidade e outliers para variáveis transacionais e de endereço
- Inspeção visual de ciclos e sazonalidades

### 4. Engenharia de Features Temporais

#### 4.1 Retornos e Volatilidade
- `volatility_rolling`: desvio padrão móvel
- `volatility_zscore`: padronização da volatilidade
- `volatility_jump_flag`: flag binária de salto

#### 4.2 Decomposição Estrutural da Série (STL)
- `seasonal_peak_month`, `seasonal_trough_month`
- `stl_spikiness`: variabilidade residual
- `stl_trend_linearity`: linearidade da tendência
- `stl_trend_curvature`: curvatura da tendência
- `stl_e_acf1`, `stl_e_acf10`: autocorrelação do resíduo

#### 4.3 Análise Espectral (FFT)
- `fft_dominant_freq`: frequência dominante
- `fft_energy_ratio`: concentração de energia espectral
- `fft_peak_amplitude`: pico de amplitude no espectro
- `fft_spectral_entropy`: entropia espectral (desordem)

#### 4.4 Detecção de Padrões Cíclicos
- `dias_desde_ultimo_vale`: recorrência de mínimos
- `duracao_ultimo_ciclo`: duração do ciclo anterior
- `amplitude_ultimo_ciclo`: amplitude detectada
- `fase_ciclo`: posição relativa no ciclo
- `flag_em_topo`: indicador binário de pico

#### 4.5 Estrutura e Persistência Temporal
- `acf_1`, `acf_5`, `acf_10`: autocorrelações simples
- `pacf_1`: autocorrelação parcial
- `ADF_stat`, `ADF_pvalue`: teste Dickey-Fuller
- `KPSS_stat`, `KPSS_pvalue`: teste KPSS
- `ndiffs_est_adf`: número estimado de diferenciações

### 5. Estacionariedade e Transformações
- Transformações aplicadas: `log`, `BoxCox`, `diff`
- Aplicação de testes estatísticos para determinar ruído branco e variância constante
- Correções de heterocedasticidade e normalização

### 6. Pré-Modelagem
- Benchmark com modelos ARIMA
- Diagnóstico dos resíduos:
  - `arima_fitted`: valor ajustado
  - `arima_resid`: erro do modelo
- Avaliação temporal da qualidade do ajuste

### 7. Consolidação das Features

**Tendência**
- `rolling_mean`, `stl_trend_linearity`, `stl_trend_curvature`

**Volatilidade**
- `volatility_rolling`, `volatility_zscore`, `fft_energy_ratio`

**Ciclicidade**
- `dias_desde_ultimo_vale`, `amplitude_ultimo_ciclo`, `flag_em_topo`, `seasonal_peak_month`

**Frequência**
- `fft_dominant_freq`, `fft_peak_amplitude`, `fft_spectral_entropy`

**Correlação**
- `acf_1`, `acf_5`, `acf_10`, `pacf_1`

**Regimes**
- `rolling_std`, `stationarity_zscore`, `regime_change`

**Outras Estruturas**
- `anova_p_mes`, `anova_p_ano`, `anova_p_interacao`: significância estatística da sazonalidade
- `stl_spikiness`, `stl_e_acf1`, `stl_e_acf10`: complexidade do resíduo

---

## Organização dos Dados

bitcoin_features/  
├── blockchain_blocks_part/  
├── blockchain_transactions_part/  
├── blockchain_addresses_part/  
├── features_temp/  
├── features_dolar_parquet/  
└── block_range_checkpoint.txt  

---

## Conclusão

Este projeto é a base para a construção de modelos preditivos robustos sobre o comportamento do Bitcoin, explorando desde a coleta de dados brutos até a engenharia de variáveis altamente informativas.

---

## Em Andamento

### XGBoost e LightGBM para Seleção de Features

- Treinamento de modelos supervisonados com validação temporal
- Avaliação da importância relativa das features criadas
- Otimização de hiperparâmetros
- Seleção automatizada de variáveis mais relevantes para previsão

### NLP - Sentimento de Mercado

Coleta e análise de dados de sentimento de fontes como:
- CryptoPanic
- Twitter e Reddit
- Notícias via Google News API
- Métricas como polaridade, score de relevância, frequência de termos

Objetivo: gerar embeddings de contexto e sentimento diário, que serão integrados ao modelo preditivo como variáveis externas de impacto.

### Modelagem com LSTM (Deep Learning)

- Modelagem de séries temporais com redes neurais recorrentes (LSTM)
- Avaliação de janelas temporais ótimas (lags)
- Inclusão de variáveis exógenas e embeddings de sentimento
- Detecção de padrões complexos e regimes de longo prazo

### Transformers Temporais
- Aplicação de architectures baseadas em Attention para séries temporais
- Comparação com LSTM em janelas multivariadas
- Estudo de comportamento da atenção sobre variáveis como volume, sentimento e preço

---

## Próximos Passos

- Validação temporal cruzada
- Deploy em ambiente monitorado
