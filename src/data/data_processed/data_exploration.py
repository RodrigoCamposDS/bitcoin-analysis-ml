import os
import pyarrow.parquet as pq
import pyarrow as pa
import asyncio
import aiohttp
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


multiprocessing.set_start_method("fork", force=True)  # --> Configura o método de multiprocessamento para MacOS / Configures multiprocessing method for MacOS <--

# URL BASE DA API DA BLOCKCHAIN
BLOCKCHAIN_API_URL = "https://api.blockchain.com/v3/blocks/"  # --> URL para API da blockchain / URL for the blockchain API <--

# CAMINHOS DAS TABELAS
PASTA_BLOCKS = "/Users/rodrigocampos/Library/Mobile Documents/com~apple~CloudDocs/requests/blockchain_blocks_part"
PASTA_TRANSACTIONS = "/Users/rodrigocampos/Library/Mobile Documents/com~apple~CloudDocs/requests/blockchain_transactions_part"
PASTA_ADDRESSES = "/Users/rodrigocampos/Library/Mobile Documents/com~apple~CloudDocs/requests/blockchain_addresses_part"

# LISTAR ARQUIVOS PARQUET
def listar_arquivos(pasta):
    """
    Lista os arquivos .parquet presentes em um diretório.
    
    Parâmetros:
    - pasta (str): Caminho para o diretório onde os arquivos .parquet estão localizados / Directory path where .parquet files are located.
    
    Retorna:
    - List[str]: Lista de caminhos completos para os arquivos .parquet / List of full file paths for .parquet files.
    """
    return [os.path.join(pasta, f) for f in os.listdir(pasta) if f.endswith(".parquet")]

# Define the path to find files
arquivos_blocks = listar_arquivos(PASTA_BLOCKS)  # --> Lista de arquivos dos blocos / List of block files <--
arquivos_transactions = listar_arquivos(PASTA_TRANSACTIONS)  # --> Lista de arquivos das transações / List of transaction files <--
arquivos_addresses = listar_arquivos(PASTA_ADDRESSES)  # --> Lista de arquivos dos endereços / List of address files <--

# DEFINIR ESQUEMAS CORRIGIDOS
# Definindo os esquemas para os arquivos Parquet / Defining schemas for the Parquet files
schema_blocks_corrigido = pa.schema([
    pa.field("block_height", pa.int64()),
    pa.field("timestamp", pa.timestamp("ms")),
    pa.field("block_hash", pa.string()),
    pa.field("num_transactions", pa.int64()),
    pa.field("total_btc_moved", pa.float64()),
    pa.field("block_reward", pa.float64()),
    pa.field("total_fees", pa.float64()),
    pa.field("block_value_btc", pa.float64()),
])

schema_transactions_corrigido = pa.schema([
    pa.field("tx_hash", pa.string()),
    pa.field("block_height", pa.int64()),          
    pa.field("timestamp", pa.timestamp("ms")),
    pa.field("num_inputs", pa.int64()),
    pa.field("num_outputs", pa.int64()),
    pa.field("total_input", pa.float64()),
    pa.field("total_output", pa.float64()),
    pa.field("fee", pa.float64()),
    pa.field("transaction_size", pa.int64()),
])

schema_addresses_corrigido = pa.schema([
    pa.field("tx_hash", pa.string()),
    pa.field("block_height", pa.int64()),
    pa.field("timestamp", pa.timestamp("ms")),
    pa.field("address", pa.string()),
    pa.field("direction", pa.string()),
    pa.field("amount", pa.float64()),
    pa.field("balance_before", pa.float64()),
    pa.field("balance_after", pa.float64()),
    pa.field("wallet_type", pa.string()),
    pa.field("is_zero", pa.bool_()),
])

# FUNÇÃO PARA VERIFICAR E CORRIGIR ARQUIVO PARQUET
def verificar_arquivo(arquivo, schema_corrigido):
    """
    Verifica se o arquivo Parquet está corrompido e tenta corrigir o esquema.

    Parâmetros:
    - arquivo (str): Caminho para o arquivo Parquet / Path to the Parquet file.
    - schema_corrigido (pa.schema): Esquema corrigido para o arquivo Parquet / Correct schema for the Parquet file.

    Retorna:
    - str: O ID do bloco ou 1 se o arquivo foi corrigido / Block ID or 1 if the file was corrected.
    """
    try:
        table = pq.read_table(arquivo)  # Lê o arquivo Parquet / Reads the Parquet file
        schema_atual = table.schema  # Obtém o esquema atual do arquivo / Gets the current schema of the file

        # Se houver campos incorretos, faz a correção / If there are incorrect fields, it will cast the schema
        if any(str(field.type) == "timestamp[ns, tz=UTC]" for field in schema_atual):
            table_corrigida = table.cast(schema_corrigido)  # Converte para o esquema corrigido / Casts to the corrected schema
            pq.write_table(table_corrigida, arquivo)  # Reescreve o arquivo Parquet com o esquema corrigido / Rewrites the Parquet file with the corrected schema
        
        return 1  # Arquivo corrigido / File corrected
    except Exception as e:
        print(f"Arquivo corrompido ou ausente: {arquivo} -> {e}")
        block_height = os.path.basename(arquivo).split("_")[-1].replace(".parquet", "")  # Extrai o ID do bloco / Extracts block ID from filename
        return block_height  # Retorna o ID do bloco / Returns the block ID

# EXECUTA A VERIFICAÇÃO DOS ARQUIVOS EM PARALELO
def encontrar_corrompidos(arquivos, schema):
    """
    Verifica e encontra arquivos corrompidos em paralelo.

    Parâmetros:
    - arquivos (list): Lista de arquivos Parquet a serem verificados / List of Parquet files to check.
    - schema (pa.schema): Esquema corrigido para os arquivos / Correct schema for the files.
    
    Retorna:
    - list: Lista de blocos corrompidos / List of corrupted blocks.
    """
    blocos_corrompidos = []
    with tqdm(total=len(arquivos), desc="Verificando arquivos", unit="arquivo") as pbar:  # Barra de progresso / Progress bar
        with ProcessPoolExecutor(max_workers=6) as executor:
            for resultado in executor.map(verificar_arquivo, arquivos, [schema] * len(arquivos)):
                if isinstance(resultado, str):
                    blocos_corrompidos.append(resultado)  # Adiciona os blocos corrompidos / Adds corrupted blocks
                pbar.update(1)  # Atualiza a barra de progresso / Updates the progress bar

    return blocos_corrompidos

# FUNÇÃO ASSÍNCRONA PARA BAIXAR ARQUIVOS CORROMPIDOS
async def baixar_arquivo_corrompido(session, block_height, pasta, pbar):
    """
    Baixa novamente o arquivo corrompido e o salva como Parquet.

    Parâmetros:
    - session (aiohttp.ClientSession): Sessão para a requisição HTTP / HTTP request session.
    - block_height (str): ID do bloco / Block ID.
    - pasta (str): Caminho da pasta onde o arquivo será salvo / Folder path to save the file.
    - pbar (tqdm): Barra de progresso / Progress bar.
    
    Retorna:
    - bool: True se o arquivo for baixado com sucesso, False caso contrário / True if the file is successfully downloaded, False otherwise.
    """
    url = f"{BLOCKCHAIN_API_URL}{block_height}"
    destino = f"{pasta}/block_{block_height}.parquet"

    for tentativa in range(3):  # Tenta até 3 vezes / Retries up to 3 times
        try:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    dados = await response.json()  # Obtém os dados do bloco / Gets the block data

                    # Converte JSON para Tabela PyArrow / Converts JSON to PyArrow Table
                    table = pa.Table.from_pydict(dados)
                    pq.write_table(table, destino)  # Salva o arquivo Parquet / Saves the Parquet file

                    print(f"Bloco {block_height} baixado e salvo: {destino}")
                    pbar.update(1)  # Atualiza a barra de progresso / Updates the progress bar
                    return True
                elif response.status in {500, 502, 503}:  # Erros temporários / Temporary errors
                    print(f"API instável para {block_height}, tentativa {tentativa+1}...")
                    await asyncio.sleep(2 ** tentativa)  # Aguarda um tempo antes de tentar novamente / Waits before retrying

        except Exception as e:
            print(f"Erro ao baixar {block_height}: {e}, tentativa {tentativa+1}")
            await asyncio.sleep(2 ** tentativa)  # Aguarda um tempo antes de tentar novamente / Waits before retrying

    print(f"Falha ao baixar {block_height} após 3 tentativas.")  # Falha ao baixar / Failed to download after 3 attempts
    return False

# EXECUTA O PROCESSO COMPLETO
async def pipeline_corrigir_blocos():
    """
    Encontra e corrige arquivos corrompidos, baixando-os novamente se necessário.

    Inicia o processo de verificação, correção e download dos arquivos Parquet corrompidos.
    """
    print("Buscando arquivos corrompidos ou ausentes...")

    # Encontra arquivos corrompidos / Finds corrupted files
    corrompidos_blocks = encontrar_corrompidos(arquivos_blocks, schema_blocks_corrigido)
    corrompidos_transactions = encontrar_corrompidos(arquivos_transactions, schema_transactions_corrigido)
    corrompidos_addresses = encontrar_corrompidos(arquivos_addresses, schema_addresses_corrigido)

    total_corrompidos = len(corrompidos_blocks) + len(corrompidos_transactions) + len(corrompidos_addresses)

    if total_corrompidos == 0:
        print("Nenhum arquivo corrompido encontrado.")  # Nenhum arquivo corrompido / No corrupted files found
        return

    print(f"{total_corrompidos} arquivos corrompidos ou ausentes detectados! Baixando novamente...")

    # Baixa os arquivos corrompidos com barra de progresso / Downloads corrupted files with a progress bar
    with tqdm(total=total_corrompidos, desc="Baixando blocos corrompidos", unit="bloco") as pbar:
        async with aiohttp.ClientSession() as session:
            tarefas = []
            for bloco_id in corrompidos_blocks:
                tarefas.append(baixar_arquivo_corrompido(session, bloco_id, PASTA_BLOCKS, pbar))
            for bloco_id in corrompidos_transactions:
                tarefas.append(baixar_arquivo_corrompido(session, bloco_id, PASTA_TRANSACTIONS, pbar))
            for bloco_id in corrompidos_addresses:
                tarefas.append(baixar_arquivo_corrompido(session, bloco_id, PASTA_ADDRESSES, pbar))

            await asyncio.gather(*tarefas)

    print("Processo finalizado!")  # Process completed / Process completed!

# Rodar pipeline no Notebook sem conflitos / Run pipeline in Notebook without conflicts
async def executar_pipeline():
    await pipeline_corrigir_blocos()

