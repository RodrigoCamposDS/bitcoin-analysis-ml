import asyncio
import aiohttp
import pandas as pd
import os
import time
from datetime import datetime
from functools import lru_cache
import os
import pandas as pd
from datetime import datetime
from datetime import datetime, timezone
import random  # Para adicionar um leve jitter no backoff
from aiohttp.client_exceptions import ClientPayloadError, ClientResponseError, ServerDisconnectedError, ClientConnectorError


PASTA_DESTINO = "/Users/rodrigocampos/Library/Mobile Documents/com~apple~CloudDocs/requests"  
# --> Defines the main directory for data storage / Define o diretório principal para armazenamento de dados <--
PASTA_PARTICOES = os.path.join(PASTA_DESTINO)  
# --> Constructs the path for partitioned storage / Constrói o caminho para o armazenamento particionado <--
os.makedirs(PASTA_PARTICOES, exist_ok=True)  
# --> Ensures the directory exists, creating it if necessary / Garante que o diretório exista, criando-o se necessário <--
BLOCKCHAIN_INFO_URL = "https://blockchain.info/latestblock"  
# --> API URL to fetch the latest block / URL da API para buscar o bloco mais recente <--
BLOCKCHAIN_TX_URL = "https://blockchain.info/rawblock/"  
# --> API base URL to fetch raw block data / URL base da API para buscar dados brutos de blocos <--
BLOCKCHAIN_TICKER_URL = "https://blockchain.info/ticker"  
# --> API URL to fetch the latest Bitcoin price / URL da API para buscar o preço mais recente do Bitcoin <--
wallet_cache = {}  
# --> Dictionary cache to store wallet types and avoid redundant computations / Dicionário cache para armazenar tipos de carteiras e evitar cálculos redundantes <--

# 🔶 ₿ ==| IDENTIFICAR_WALLET |============= ₿ ============== | IDENTIFICAR_WALLET |============= ₿ ================| IDENTIFICAR_WALLET |============= ₿ ===========| IDENTIFICAR_WALLET |============ ₿ ===========| IDENTIFICAR_WALLET |===========================


def identificar_tipo_wallet(endereco):  
    # --> Determines wallet type based on address prefix / Determina o tipo de carteira com base no prefixo do endereço <--
    if endereco in wallet_cache:
        return wallet_cache[endereco]  
        # --> Returns cached result if available / Retorna o resultado armazenado em cache, se disponível <--
    if endereco.startswith("1"):
        tipo = "P2PKH (Legacy)"  
        # --> Legacy wallet format (Pay-to-PubKey-Hash) / Formato de carteira legado (Pagar para Hash de Chave Pública) <--
    elif endereco.startswith("3"):
        tipo = "P2SH (Multisig ou SegWit Antigo)"  
        # --> Pay-to-Script-Hash, used for multisig or older SegWit addresses / Pagar para Script Hash, usado para multisig ou SegWit antigo <--
    elif endereco.startswith("bc1q"):
        tipo = "Bech32 (SegWit Nativo)"  
        # --> Native SegWit format with lower fees / Formato SegWit Nativo com taxas menores <--
    elif endereco.startswith("bc1p"):
        tipo = "Taproot"  
        # --> Taproot-enabled address, improving privacy and efficiency / Endereço compatível com Taproot, melhorando privacidade e eficiência <--
    else:
        tipo = "Desconhecido"  
        # --> Unknown address type / Tipo de endereço desconhecido <--
    wallet_cache[endereco] = tipo  
    # --> Stores result in cache to optimize future lookups / Armazena o resultado no cache para otimizar futuras consultas <--
    return tipo  


# 🔶 ₿ ==| INFERIR_WALLET |============= ₿ ============== | INFERIR_WALLET |============== ₿ ===============| INFERIR_WALLET |============== ₿ ==========| INFERIR_WALLET |================ ₿ =============| INFERIR_WALLET |============== ₿ ============| INFERIR_WALLET |=


@lru_cache(maxsize=10_000)  
# --> Applies LRU caching to optimize repeated wallet type inferences / Aplica cache LRU para otimizar inferências repetidas de tipos de carteira <--
def inferir_tipo_wallet(num_inputs, num_outputs, total_input):  
    # --> Infers wallet type based on transaction behavior / Infere o tipo de carteira com base no comportamento da transação <--
    if num_inputs > 10 and num_outputs > 10:
        return "Exchange"  
        # --> High input/output count suggests exchange transactions / Alto número de entradas/saídas sugere transações de exchange <--
    elif num_inputs == 1 and num_outputs == 1:
        return "Carteira Pessoal"  
        # --> Single input/output indicates a personal wallet / Entrada/saída única indica carteira pessoal <--
    elif num_inputs > 2 and num_outputs > 1:
        return "Multisig"  
        # --> Multiple inputs and outputs suggest a multi-signature wallet / Múltiplas entradas e saídas sugerem uma carteira multiassinatura <--
    elif total_input > 100:
        return "Whale (Grande Investidor)"  
        # --> Transactions exceeding 100 BTC indicate large investors / Transações acima de 100 BTC indicam grandes investidores <--    
    return "Desconhecido"  
    # --> Default case for unclassified wallets / Caso padrão para carteiras não classificadas <--


# 🔶 ₿ ==| FETCH_BLOC |============= ₿ ============== | FETCH_BLOC |================ ₿ =============| FETCH_BLOC |============ ₿ ============| FETCH_BLOC |=========== ₿ ============| FETCH_BLOC |============= ₿ ==============| FETCH_BLOC |================


async def fetch_block(session, height, retries=5, base_delay=2):
    """Busca um bloco específico da blockchain com retry e exponential backoff."""

    url = f"https://blockchain.info/rawblock/{height}"
    # --> Monta a URL de requisição para o bloco específico / Builds request URL for the specific block <--

    for tentativa in range(retries):
        # --> Loop de tentativas com número máximo definido / Retry loop with defined maximum attempts <--
        try:
            async with session.get(url, timeout=60) as resp:
                # --> Envia requisição HTTP assíncrona com timeout de 60 segundos / Sends async HTTP request with 60s timeout <--

                if resp.status == 200:
                    return await resp.json()
                    # --> Retorna o JSON do bloco se a resposta for bem-sucedida / Returns JSON if request is successful <--

                elif resp.status in {400, 500, 502, 503}:
                    print(f"Erro no bloco {height}, tentativa {tentativa + 1} de {retries}. Retentando...")
                    await asyncio.sleep(base_delay * (2 ** tentativa) + random.uniform(0, 1))
                    # --> Espera com backoff exponencial e jitter para evitar sobrecarga / Waits using exponential backoff and jitter <--

        except (asyncio.TimeoutError, ClientPayloadError, ClientResponseError, ServerDisconnectedError, ClientConnectorError) as e:
            print(f"Erro no bloco {height}: {str(e)}. Tentativa {tentativa + 1} de {retries}. Retentando...")
            await asyncio.sleep(base_delay * (2 ** tentativa) + random.uniform(0, 1))
            # --> Trata erros de rede e aplica backoff antes de nova tentativa / Handles network errors and applies backoff <--

    print(f"Falha ao obter bloco {height} após {retries} tentativas.")
    return None
    # --> Retorna None se todas as tentativas falharem / Returns None if all retries fail <--

# 🔶 ₿ ==| CALCULAR_RECOMPENSA_BLOCO |============== ₿ ============= | CALCULAR_RECOMPENSA_BLOCO |============== ₿ ===============| CALCULAR_RECOMPENSA_BLOCO |============= ₿ ===========| CALCULAR_RECOMPENSA_BLOCO |=========== ₿ ===========| BUSCAR_PRECOS_BITCOIN |=====


# Função para determinar a recompensa do bloco com base no halving
def calcular_recompensa_bloco(block_height):
    """
    Retorna a recompensa do bloco em BTC com base na altura do bloco e nos halvings.
    """
    halvings = block_height // 210000  # A cada 210.000 blocos, o halving ocorre
    recompensa_inicial = 50  # Começou com 50 BTC
    return recompensa_inicial / (2 ** halvings)  # Divide pela quantidade de halvings já ocorridos


# 🔶 ₿ ==| CALCULAR_VALORES_BLOCO |============ ₿ =============== | CALCULAR_VALORES_BLOCO |============== ₿ ===============| CALCULAR_VALORES_BLOCO |============ ₿ ============| CALCULAR_VALORES_BLOCO |============= ₿ ===========| CALCULAR_VALORES_BLOCO |============= ₿ ===========| MAIN |============= ₿ ===========| MAIN |============= ₿ ===========| MAIN |======


def calcular_valores_bloco(bloco):
    """
    Calcula:
    - Total de BTC movimentado no bloco.
    - Total de taxas pagas aos mineradores.
    - Recompensa do bloco com base no halving.
    - Valor total do bloco em BTC (recompensa + taxas).
    - Valor puro do bloco (somente recompensa + taxas, sem BTC de transações).
    """

    total_btc_moved = sum(out["value"] for tx in bloco.get("tx", []) for out in tx.get("out", [])) / 1e8  
    # --> Soma o valor de todas as saídas (outputs) de todas as transações do bloco, convertido de satoshis para BTC /
    # --> Sums all transaction output values from the block, converting from satoshis to BTC <--

    total_fees = sum(tx.get("fee", 0) for tx in bloco.get("tx", [])) / 1e8  
    # --> Soma todas as taxas individuais das transações, também convertidas para BTC /
    # --> Sums all transaction fees, also converted to BTC <--

    recompensa_bloco = calcular_recompensa_bloco(bloco["height"]) 
    # --> Calcula a recompensa base do bloco com base na altura (considerando halving) /
    # --> Calculates the block reward based on height (taking halving into account) <--

    valor_total_bloco = recompensa_bloco + total_fees  
    # --> Soma da recompensa com as taxas — valor total ganho pelos mineradores /
    # --> Total miner reward = block reward + transaction fees <--

    return total_btc_moved, total_fees, recompensa_bloco, valor_total_bloco
    # --> Retorna os valores calculados: movimentação total, taxas, recompensa e valor total do bloco /
    # --> Returns calculated values: total moved BTC, fees, reward, and total block value <--


# 🔶 ₿ ==| buscar_blocos |============= ₿ ============== | buscar_blocos |=============== ₿ ==============| buscar_blocos |============ ₿ ============| buscar_blocos |=============== ₿ ================| buscar_blocos |=============| buscar_blocos |============ ₿ ============| buscar_blocos |=============== ₿ ================| buscar_blocos |========

async def buscar_blocos(quantidade=100, modo="recente", bloco_inicio=None, bloco_fim=None):
    """
    Busca blocos da blockchain de forma assíncrona com base em quatro modos:
    - modo="recente": busca blocos mais novos a partir do último salvo.
    - modo="antigo": busca blocos mais antigos antes do mais antigo salvo.
    - modo="manual": permite definir intervalo com bloco_inicio e bloco_fim.
    - modo="auto": busca blocos do topo mais recente da blockchain para trás, ideal para primeira execução.
    """
    assert modo in {"recente", "antigo", "manual", "auto"}, "Modo deve ser 'recente', 'antigo', 'manual' ou 'auto'."
    # --> Valida o modo de operação fornecido / Validates the provided operation mode <--

    if modo == "manual":
        assert bloco_inicio is not None and bloco_fim is not None, "Defina bloco_inicio e bloco_fim para o modo manual."
        # --> Garante que o intervalo seja definido no modo manual / Ensures block range is defined in manual mode <--
        blocos_para_baixar = list(range(bloco_inicio, bloco_fim + 1))
        # --> Cria a lista de blocos a partir do intervalo fornecido / Creates the list of blocks from the defined range <--
        blocos_para_baixar = [b for b in blocos_para_baixar if b > 0]
        # --> Remove blocos inválidos (menores ou iguais a zero) / Removes invalid blocks (less than or equal to zero) <--

    else:
        async with aiohttp.ClientSession() as session:
            async with session.get(BLOCKCHAIN_INFO_URL) as resp:
                topo = (await resp.json())["height"]
        # --> Recupera a altura mais recente da blockchain / Retrieves the most recent block height from the blockchain <--

        if modo == "auto":
            blocos_para_baixar = list(range(topo - quantidade + 1, topo + 1))
            # --> Define blocos mais recentes para a primeira execução / Sets most recent blocks for first-time execution <--
            blocos_para_baixar = [b for b in blocos_para_baixar if b > 0]
            # --> Filtra blocos válidos / Filters valid blocks <--

        elif modo == "recente":
            bloco_referencia = get_checkpoint("recente")
            # --> Obtém o último bloco salvo (mais recente) / Gets the last saved block (most recent) <--
            if bloco_referencia is None:
                print("[INFO] Nenhum checkpoint encontrado. Use 'auto' para iniciar.")
                return []
            blocos_para_baixar = list(range(bloco_referencia + 1, bloco_referencia + quantidade + 1))
            # --> Define os próximos blocos após o último salvo / Sets the next blocks after the last saved <--
            blocos_para_baixar = [b for b in blocos_para_baixar if b <= topo]
            # --> Garante que os blocos não excedam o topo da blockchain / Ensures blocks don’t exceed current blockchain top <--

        elif modo == "antigo":
            bloco_referencia = get_checkpoint("antigo")
            # --> Obtém o bloco mais antigo salvo / Gets the earliest saved block <--
            if bloco_referencia is None:
                print("[INFO] Nenhum checkpoint encontrado. Use 'auto' para iniciar.")
                return []
            blocos_para_baixar = list(range(bloco_referencia - quantidade, bloco_referencia))
            # --> Define blocos anteriores ao mais antigo salvo / Sets blocks prior to the earliest saved <--
            blocos_para_baixar = [b for b in blocos_para_baixar if b > 0]
            # --> Remove blocos inválidos (menores ou iguais a zero) / Removes invalid blocks (<= 0) <--

    if not blocos_para_baixar:
        print("[INFO] Nenhum bloco válido disponível para baixar.")
        return []
    # --> Interrompe a execução se não houver blocos válidos / Halts execution if no valid blocks available <--

    print(f"[INFO] Buscando {len(blocos_para_baixar)} blocos (modo={modo})...")
    # --> Exibe a quantidade de blocos que serão buscados / Logs the number of blocks to be fetched <--

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_block(session, height) for height in blocos_para_baixar]
        # --> Cria lista de tarefas assíncronas para buscar blocos / Creates list of async tasks to fetch blocks <--
        return await asyncio.gather(*tasks)
        # --> Executa todas as tarefas de forma concorrente / Executes all tasks concurrently <--


# 🔶 ₿ ==| SALVAR_DADOS_PARQUET |============= ₿ ============= | SALVAR_DADOS_PARQUET |=============== ₿ ==============| SALVAR_DADOS_PARQUET |============= ₿ ===========| SALVAR_DADOS_PARQUET |============ ₿ ===========| SALVAR_DADOS_PARQUET |=======================


def salvar_dados_parquet(df, pasta, prefixo_arquivo):  
# --> Saves a DataFrame as partitioned Parquet files / Salva um DataFrame como arquivos Parquet particionados <--
    if df.empty:  
        return  
     # --> Returns early if the DataFrame is empty / Retorna imediatamente se o DataFrame estiver vazio <--
    caminho_pasta = os.path.join(PASTA_PARTICOES, pasta)  
    os.makedirs(caminho_pasta, exist_ok=True)  
    # --> Creates the target directory if it doesn't exist / Cria o diretório de destino se não existir <--
    for block_value in df["block_height"].unique():  
    # --> Iterates over unique block heights in the DataFrame / Itera sobre os blocos únicos no DataFrame <--
        df_part = df[df["block_height"] == block_value]  
        caminho_arquivo = os.path.join(caminho_pasta, f"{prefixo_arquivo}_{block_value}.parquet")  
        df_part.to_parquet(caminho_arquivo, index=False, engine="pyarrow")  
        # --> Saves each block as an individual Parquet file / Salva cada bloco como um arquivo Parquet individual <--
    print(f"Arquivos particionados salvos em: {caminho_pasta}")  
    # --> Prints the save location of partitioned files / Imprime o local onde os arquivos particionados foram salvos <--


# 🔶 ₿ ==| SALVAR_BLOCOS_PARQUET |============ ₿ ============== | SALVAR_BLOCOS_PARQUET |============= ₿ ===============| SALVAR_BLOCOS_PARQUET |============ ₿ ============| SALVAR_BLOCOS_PARQUET |=============== ₿ ============| SALVAR_BLOCOS_PARQUET |

MAIS_ANTIGO_BLOCO_PATH = os.path.join(PASTA_PARTICOES, "mais_antigo_bloco.txt")  
# --> Define o caminho para armazenar o último bloco salvo (menor altura) / Path to store the last (oldest) saved block <--

MAIS_RECENTE_BLOCO_PATH = os.path.join(PASTA_PARTICOES, "mais_recente_bloco.txt")  
# --> Define o caminho para armazenar o bloco mais recente salvo / Path to store the most recent saved block <--


async def salvar_blocos_parquet(blocos, tipo):
    # --> tipo = "recente" para salvar blocos mais novos / "antigo" para blocos anteriores <--

    if not blocos:  
        print("Nenhum bloco novo encontrado para salvar.")  
        # --> Retorna imediatamente se não houver blocos novos / Returns early if no new blocks are found <--
        return  

    caminho_pasta_blocos = os.path.join(PASTA_PARTICOES, "blockchain_blocks_part")  
    os.makedirs(caminho_pasta_blocos, exist_ok=True)  
    # --> Cria o diretório de destino se ele ainda não existir / Creates the target directory if it doesn't exist <--

    blocos_processados = []  
    # --> Lista para armazenar os blocos que foram processados com sucesso / List to store successfully processed blocks <--

    async with aiohttp.ClientSession() as session:  
        # --> Inicia uma sessão HTTP assíncrona (reservado para uso futuro) / Starts an async HTTP session (reserved for future use) <--
        for bloco in blocos:  
            # --> Itera sobre todos os blocos recebidos / Iterates over all received blocks <--
            bloco_id = bloco["height"]  
            # --> Extrai o número (altura) do bloco / Extracts the block height <--

            caminho_arquivo = os.path.join(caminho_pasta_blocos, f"block_{bloco_id}.parquet")  
            # --> Define o caminho completo para salvar o bloco como arquivo Parquet / Defines the full path for saving the block as Parquet file <--

            if os.path.exists(caminho_arquivo):  
                continue  
                # --> Pula a iteração se o arquivo do bloco já existir / Skips if the block file already exists <--

            timestamp_bloco = bloco["time"]  
            # --> Obtém o timestamp do bloco em formato UNIX / Gets the block timestamp in UNIX format <--

            total_btc_moved, total_fees, recompensa_bloco, valor_total_bloco = calcular_valores_bloco(bloco)  
            # --> Calcula os valores financeiros do bloco em BTC / Calculates the financial values of the block in BTC <--

            df = pd.DataFrame([{  
                "block_height": bloco_id,  
                "timestamp": datetime.fromtimestamp(float(timestamp_bloco), tz=timezone.utc),  
                # --> Converte o timestamp UNIX para formato datetime com timezone UTC / Converts UNIX timestamp to datetime with UTC timezone <--
                "block_hash": bloco["hash"],  
                "num_transactions": len(bloco["tx"]),  
                # --> Conta o número de transações no bloco / Counts the number of transactions in the block <--
                "total_btc_moved": total_btc_moved,  
                # --> BTC total movimentado no bloco / Total BTC moved in the block <--
                "block_reward": recompensa_bloco,  
                # --> Recompensa recebida pelo minerador / Miner reward for the block <--
                "total_fees": total_fees,  
                # --> Taxas pagas pelos usuários nesse bloco / User fees in the block <--
                "block_value_btc": valor_total_bloco  
                # --> Valor total do bloco (recompensa + taxas) / Total block value (reward + fees) <--
            }])  

            df.to_parquet(caminho_arquivo, index=False, engine="pyarrow")  
            # --> Salva os dados do bloco como arquivo Parquet / Saves the block data as a Parquet file <--
            blocos_processados.append(bloco_id)  
            # --> Armazena o ID do bloco processado / Stores the processed block ID <--

    if blocos_processados:
        if tipo == "recente":
            with open(MAIS_RECENTE_BLOCO_PATH, "w") as f:
                f.write(str(max(blocos_processados)))
        elif tipo == "antigo":
            with open(MAIS_ANTIGO_BLOCO_PATH, "w") as f:
                f.write(str(min(blocos_processados)))
        elif tipo == "auto":
            # Salva ambos os arquivos para iniciar os checkpoints
            with open(MAIS_RECENTE_BLOCO_PATH, "w") as f:
                f.write(str(max(blocos_processados)))
            with open(MAIS_ANTIGO_BLOCO_PATH, "w") as f:
                f.write(str(min(blocos_processados)))

    novos_blocos = len(blocos_processados)
    blocos_totais = len(os.listdir(caminho_pasta_blocos))  
    print(f"\n-> Blocos:")
    print(f"{novos_blocos} Blocos salvos agora em --------------------------------------> {caminho_pasta_blocos}")
    print(f"{blocos_totais} Blocos acumulados no total em blockchain_blocks_part")
    # --> Imprime resumo da operação no terminal / Prints summary of the operation in the terminal <--


# 🔶 ₿ ==| GET_CHECKPOINT |============= ₿ ============== | GET_CHECKPOINT |============= ₿ ==============| GET_CHECKPOINT |============= ₿ ==============| GET_CHECKPOINT |============= ₿ ==============


def get_checkpoint(tipo):
    """
    Retorna a altura de bloco mais confiável, com base nos arquivos `.parquet` e no `.txt`.  
    Atualiza o arquivo `.txt` automaticamente se estiver desatualizado.

    Returns the most reliable block height based on `.parquet` files and `.txt`.  
    Automatically updates the `.txt` file if it's out of sync.
    """
    assert tipo in {"antigo", "recente"}, "Tipo deve ser 'antigo' ou 'recente' / Type must be 'antigo' or 'recente'"

    # → Define o caminho do arquivo de referência (.txt) / Defines the reference file path (.txt)
    caminho_txt = os.path.join(PASTA_PARTICOES, f"mais_{tipo}_bloco.txt")

    # → Define a pasta onde estão os blocos salvos / Defines the folder where the blocks are saved
    caminho_pasta_blocos = os.path.join(PASTA_PARTICOES, "blockchain_blocks_part")

    valor_txt = None
    valor_pasta = None

    # → Tenta ler o valor do arquivo .txt, se existir / Attempts to read from .txt if it exists
    if os.path.exists(caminho_txt):
        with open(caminho_txt, "r") as f:
            valor_txt = int(f.read().strip())

    # → Lê todos os blocos .parquet da pasta e extrai os IDs / Reads all .parquet files and extracts block IDs
    if os.path.exists(caminho_pasta_blocos):
        arquivos = os.listdir(caminho_pasta_blocos)
        blocos_salvos = [
            int(arq.replace("block_", "").replace(".parquet", ""))
            for arq in arquivos
            if arq.startswith("block_") and arq.endswith(".parquet")
        ]
        if blocos_salvos:
            valor_pasta = min(blocos_salvos) if tipo == "antigo" else max(blocos_salvos)

    # → Define o valor final de referência com base no que estiver disponível / Defines the final reference value
    if valor_txt is not None and valor_pasta is not None:
        valor_final = min(valor_txt, valor_pasta) if tipo == "antigo" else max(valor_txt, valor_pasta)
    elif valor_txt is not None:
        valor_final = valor_txt
    elif valor_pasta is not None:
        valor_final = valor_pasta
    else:
        return None  # → Nenhum dado disponível / No data available

    # → Atualiza o arquivo .txt se necessário / Updates the .txt file if needed
    if valor_final != valor_txt:
        with open(caminho_txt, "w") as f:
            f.write(str(valor_final))

    return valor_final

  

# # 🔶 ₿ ==| SALVAR_TRANSACOES_PARQUET |============== ₿ ============= | SALVAR_TRANSACOES_PARQUET |=============== ₿ ==============| SALVAR_TRANSACOES_PARQUET |============= ₿ ===========| SALVAR_TRANSACOES_PARQUET |======================== ₿ ===========| SALVAR_TRANSACOES_PARQUET |


def salvar_transacoes_parquet(blocos):  
# --> Saves blockchain transactions as partitioned Parquet files / Salva transações da blockchain como arquivos Parquet particionados <--
    if not blocos:  
        print("Nenhuma transação nova encontrada para salvar.")  
        return  
    # --> Returns early if there are no new transactions to save / Retorna imediatamente se não houver transações novas para salvar <--
    caminho_pasta_transacoes = os.path.join(PASTA_PARTICOES, "blockchain_transactions_part")  
    os.makedirs(caminho_pasta_transacoes, exist_ok=True)  
    # --> Ensures the directory exists for storing transactions / Garante que o diretório para armazenar transações exista <--
    novas_transacoes = 0  

    for bloco in blocos:  
        bloco_id = bloco["height"]
        timestamp_bloco = bloco["time"]   
        caminho_arquivo = os.path.join(caminho_pasta_transacoes, f"transactions_{bloco_id}.parquet")  
    # --> Defines the path for the block's transaction Parquet file / Define o caminho para o arquivo Parquet das transações do bloco <--
        if os.path.exists(caminho_arquivo):  
            continue  
        # --> Skips saving if the transaction file already exists / Pula o salvamento se o arquivo de transações já existir <--
        transacoes = [  
            {  
                "tx_hash": tx["hash"],  
                "block_height": bloco_id,  
                "timestamp": max(datetime.fromtimestamp(float(tx["time"]), tz=timezone.utc), datetime.fromtimestamp(float(timestamp_bloco), tz=timezone.utc)),  
                "num_inputs": len(tx.get("inputs", [])),  
                "num_outputs": len(tx.get("out", [])),  
                "total_input": sum(inp["prev_out"]["value"] / 1e8 for inp in tx["inputs"]) if "inputs" in tx else 0,  
                "total_output": sum(out["value"] / 1e8 for out in tx["out"]) if "out" in tx else 0,  
                "fee": max(0, (sum(inp.get("prev_out", {}).get("value", 0) for inp in tx.get("inputs", [])) - sum(out.get("value", 0) for out in tx.get("out", []))) / 1e8) if tx.get("inputs") and tx.get("out") else 0,  
                "transaction_size": tx.get("size", 0)  
            }  
            for tx in bloco.get("tx", [])  
        ]  
        # --> Extracts transaction details from the block / Extrai os detalhes das transações do bloco <--
        if transacoes:  
            pd.DataFrame(transacoes).to_parquet(caminho_arquivo, index=False, engine="pyarrow")  
            # --> Saves transactions as a Parquet file / Salva as transações como um arquivo Parquet <--
            novas_transacoes += len(transacoes)  
            # --> Updates the count of new saved transactions / Atualiza a contagem de novas transações salvas <--
    print(f"\n-> Transações:")  
    print(f"{novas_transacoes} Transações salvas agora em --------------------------------------------> {caminho_pasta_transacoes}") 
    

# 🔶 ₿ ==| SALVAR_ENDERECOS_PARQUET |============= ₿ ============= | SALVAR_ENDERECOS_PARQUET |============== ₿ ===============| SALVAR_ENDERECOS_PARQUET |============= ₿ ===========| SALVAR_ENDERECOS_PARQUET |============= ₿ ===========| SALVAR_ENDERECOS_PARQUET |=


def salvar_enderecos_parquet(blocos):  
# --> Saves wallet addresses involved in blockchain transactions as partitioned Parquet files / Salva endereços de carteiras envolvidos nas transações da blockchain como arquivos Parquet particionados <--
    if not blocos:  
        print("Nenhum endereço novo encontrado para salvar.")  
        return  
    # --> Returns early if there are no new addresses to save / Retorna imediatamente se não houver endereços novos para salvar <--
    caminho_pasta_enderecos = os.path.join(PASTA_PARTICOES, "blockchain_addresses_part")  
    os.makedirs(caminho_pasta_enderecos, exist_ok=True)  
    # --> Ensures the directory exists for storing wallet addresses / Garante que o diretório para armazenar endereços de carteiras exista <--
    saldo_carteiras = {}  
    novos_enderecos = 0  

    for bloco in blocos:  
        bloco_id = bloco["height"]
        timestamp_bloco = bloco["time"]  
        caminho_arquivo = os.path.join(caminho_pasta_enderecos, f"addresses_{bloco_id}.parquet")  
    # --> Defines the path for the block's wallet address Parquet file / Define o caminho para o arquivo Parquet dos endereços do bloco <--
        if os.path.exists(caminho_arquivo):  
            continue  
        # --> Skips saving if the file already exists / Pula o salvamento se o arquivo já existir <--
        enderecos = []  

        for tx in bloco.get("tx", []):  
            timestamp_tx = max(datetime.fromtimestamp(float(tx["time"]), tz=timezone.utc), datetime.fromtimestamp(float(timestamp_bloco), tz=timezone.utc))  
            tx_hash = tx["hash"]    
            block_height = bloco["height"]  
        # --> Extracts basic transaction metadata / Extrai metadados básicos da transação <--
            heuristica_wallet = inferir_tipo_wallet(  
                len(tx.get("inputs", [])),  
                len(tx.get("out", [])),  
                sum(inp["prev_out"]["value"] / 1e8 for inp in tx["inputs"]) if "inputs" in tx else 0  
            )  
            # --> Infers the wallet type based on transaction behavior / Infere o tipo de carteira com base no comportamento da transação <--
            for entrada in tx.get("inputs", []):  
                if "prev_out" in entrada and "addr" in entrada["prev_out"]:  
                    addr = entrada["prev_out"]["addr"]  
                    amount = entrada["prev_out"]["value"] / 1e8                
                    balance_before = saldo_carteiras.get(addr, 0)  # Captura o saldo antes da transação
                    saldo_carteiras[addr] = balance_before + amount  # Atualiza o saldo da carteira somando o valor
                    balance_after = saldo_carteiras[addr]  # Captura o saldo depois da transação

                    wallet_type = wallet_cache.get(addr, identificar_tipo_wallet(addr))
 
            # --> Uses cached wallet type if available, otherwise determines it / Usa o tipo de carteira em cache, se disponível; caso contrário, determina-o <--
                    enderecos.append({  
                        "tx_hash": tx_hash,  
                        "block_height": block_height,  
                        "timestamp": timestamp_tx,  
                        "address": addr,  
                        "direction": "input",  
                        "amount": amount,  
                        "balance_before": balance_before,  
                        "balance_after": balance_after,  
                        "wallet_type": heuristica_wallet if heuristica_wallet != "Desconhecido" else wallet_type,  
                        "is_zero": balance_after == 0  
                    })  
                    # --> Stores input transaction details / Armazena os detalhes das transações de entrada <--
            for saida in tx.get("out", []):  
                    if "addr" in saida:  
                        addr = saida["addr"]  
                        amount = saida["value"] / 1e8   
                        balance_before = saldo_carteiras.get(addr, 0)  # Captura o saldo antes da transação
                        saldo_carteiras[addr] = balance_before - amount  # Atualiza o saldo da carteira subtraindo o valor
                        balance_after = max(saldo_carteiras[addr], 0)  # Garante que não haja saldo negativo

                        wallet_type = wallet_cache.get(addr, identificar_tipo_wallet(addr))

                        enderecos.append({  
                            "tx_hash": tx_hash,  
                            "block_height": block_height,  
                            "timestamp": timestamp_tx,  
                            "address": addr,  
                            "direction": "output",  
                            "amount": amount,  
                            "balance_before": balance_before,  
                            "balance_after": balance_after,  
                            "wallet_type": heuristica_wallet if heuristica_wallet != "Desconhecido" else wallet_type,  
                            "is_zero": balance_after == 0  
                        })  
                # --> Stores output transaction details / Armazena os detalhes das transações de saída <--

        if enderecos:  
            pd.DataFrame(enderecos).to_parquet(caminho_arquivo, index=False, engine="pyarrow")  
            novos_enderecos += len(enderecos)  
        # --> Saves wallet address data as a Parquet file and updates count / Salva os dados dos endereços de carteira como um arquivo Parquet e atualiza a contagem <--

    print(f"\n-> Endereços:")  
    print(f"{novos_enderecos} Endereços salvos agora em --------------------------------------------> {caminho_pasta_enderecos}")  
  

# 🔶 ₿ ==| COLETAR_DADOS_EM_LOTES |============== ₿ ============= | COLETAR_DADOS_EM_LOTES |============== ₿ ===============| COLETAR_DADOS_EM_LOTES |============ ₿ ============| COLETAR_DADOS_EM_LOTES |============ ₿ ============| COLETAR_DADOS_EM_LOTES |============ ₿ ============| COLETAR_DADOS_EM_LOTES |

async def coletar_dados_em_lotes(total_blocos, tamanho_lote, modo, bloco_inicio=None, bloco_fim=None):
    """
    Coleta dados da blockchain em lotes com base no modo especificado:
    - modo="recente": a partir do último bloco salvo.
    - modo="antigo": blocos anteriores ao mais antigo salvo.
    - modo="manual": entre bloco_inicio e bloco_fim.
    - modo="auto": do topo da blockchain para trás (sem checkpoints).
    """
    inicio = time.time()
    # --> Marca o tempo de início da coleta / Marks the start time of the collection <--

    blocos_baixados = 0
    # --> Inicializa o contador de blocos baixados / Initializes counter for downloaded blocks <--

    while blocos_baixados < total_blocos:
        # --> Loop principal que continua até atingir o número total de blocos / Main loop until total blocks reached <--

        print(f"\n-> Iniciando coleta do lote {blocos_baixados // tamanho_lote + 1} de {total_blocos // tamanho_lote}...")
        # --> Exibe o progresso da coleta em lotes / Displays progress of batch collection <--

        if modo == "manual":
            # --> Cálculo do número restante de blocos no modo manual / Calculates remaining blocks in manual mode <--
            qtde_restante = min(tamanho_lote, bloco_fim - bloco_inicio + 1 - blocos_baixados)
            if qtde_restante <= 0:
                break
            blocos = await buscar_blocos(
                qtde_restante,
                modo=modo,
                bloco_inicio=bloco_inicio + blocos_baixados,
                bloco_fim=bloco_inicio + blocos_baixados + qtde_restante - 1
            )
            # --> Busca blocos dentro do intervalo definido manualmente / Fetches blocks within defined manual range <--
        else:
            blocos = await buscar_blocos(quantidade=tamanho_lote, modo=modo)
            # --> Busca blocos com base no modo (recente, antigo ou auto) / Fetches blocks according to mode <--

        blocos = [b for b in blocos if b]
        # --> Remove blocos nulos ou com erro / Removes null or failed blocks <--

        if blocos:
            await salvar_blocos_parquet(blocos, tipo=modo)
            # --> Salva os dados dos blocos em arquivos Parquet / Saves block data to Parquet files <--

            salvar_transacoes_parquet(blocos)
            # --> Salva as transações dos blocos / Saves transactions from blocks <--

            salvar_enderecos_parquet(blocos)
            # --> Salva os endereços envolvidos nas transações / Saves addresses from transactions <--

            blocos_baixados += len(blocos)
            # --> Atualiza o contador de blocos baixados / Updates block counter <--
        else:
            print("[INFO] Nenhum bloco novo encontrado. Encerrando coleta.")
            # --> Interrompe o loop se não houver blocos válidos / Breaks the loop if no valid blocks <--
            break

        await asyncio.sleep(2)
        # --> Aguarda para evitar sobrecarga da API / Waits to avoid API overload <--

    print(f"\n-> Coleta concluída em {time.time() - inicio:.2f} segundos.")
    # --> Exibe o tempo total da coleta / Displays total collection time <--

# 🔶 ₿ ==| executar_pipeline_blockchain |============== ₿ ============= | executar_pipeline_blockchain |============== ₿ ===============| executar_pipeline_blockchain |============ ₿ ============| executar_pipeline_blockchain |============ ₿ ============| executar_pipeline_blockchain |============ ₿ ============| executar_pipeline_blockchain |

async def executar_pipeline_blockchain(
    total_blocos=1000,
    tamanho_lote=100,
    modo="antigo",
    bloco_inicio=None,
    bloco_fim=None
):
    await coletar_dados_em_lotes(
        total_blocos=total_blocos,
        tamanho_lote=tamanho_lote,
        modo=modo,
        bloco_inicio=bloco_inicio,
        bloco_fim=bloco_fim
    )
    print("\n-> Execução finalizada!")

