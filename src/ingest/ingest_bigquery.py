from google.cloud import bigquery
import os
import pandas as pd
import glob

# Configurar credenciais da conta de serviço / Configure the service account credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/rodrigocampos/Documents/Bitcoin/btcanalytics-453021-dc83ec3411cb.json"
client = bigquery.Client(project="btcanalytics-453021")  # Cria o cliente BigQuery / Creates the BigQuery client

# Informações do dataset e das tabelas no BigQuery / Dataset and table information in BigQuery
DATASET_ID = "dataset_cripto"
TABELAS_PARQUET = {
    "blockchain_blocks": "blockchain_blocks_part",  # Tabela de blocos / Blocks table
    "blockchain_transactions": "blockchain_transactions_part",  # Tabela de transações / Transactions table
    "blockchain_addresses": "blockchain_addresses_part"  # Tabela de endereços / Addresses table
}

# Caminho principal onde os arquivos Parquet são salvos / Main path where the Parquet files are stored
CAMINHO_PARTICOES = "/Users/rodrigocampos/Library/Mobile Documents/com~apple~CloudDocs/requests"

def enviar_parquet_para_bigquery(tabela_bigquery: str, pasta_parquet: str):
    """
    Lê arquivos Parquet de uma pasta local e insere os dados na tabela correspondente do BigQuery.
    
    Parâmetros:
    - tabela_bigquery (str): Nome da tabela no BigQuery onde os dados serão inseridos / BigQuery table name to insert data.
    - pasta_parquet (str): Nome da pasta onde os arquivos Parquet estão localizados / Folder name where Parquet files are stored.
    """
    caminho_completo = os.path.join(CAMINHO_PARTICOES, pasta_parquet)  # Caminho completo da pasta Parquet / Full path to Parquet folder
    arquivos_parquet = glob.glob(f"{caminho_completo}/*.parquet")  # Lista de arquivos Parquet / List of Parquet files
    total_enviados = 0  # Inicializa contador de registros enviados / Initializes the sent records counter

    for arquivo in arquivos_parquet:
        df = pd.read_parquet(arquivo)  # Lê o arquivo Parquet / Reads the Parquet file
        if df.empty:  # Se o arquivo estiver vazio, pula o processamento / If the file is empty, skip it
            continue

        registros = df.to_dict(orient="records")  # Converte o DataFrame em lista de dicionários / Converts DataFrame to list of dictionaries
        tabela_destino = f"{DATASET_ID}.{tabela_bigquery}"  # Define o nome da tabela de destino / Defines the destination table name

        errors = client.insert_rows_json(tabela_destino, registros)  # Envia os registros para BigQuery / Sends the records to BigQuery
        if errors:
            print(f"Erro ao enviar {arquivo} para {tabela_destino}:", errors)  # Se houver erro, exibe mensagem / If there's an error, print the message
        else:
            print(f"{len(registros)} registros enviados de {arquivo} → {tabela_destino}")  # Exibe o número de registros enviados / Displays the number of records sent
            total_enviados += len(registros)  # Atualiza o contador de registros / Updates the sent records counter

    print(f"\nTotal de registros enviados para {tabela_bigquery}: {total_enviados}")  # Exibe o total de registros enviados / Displays total records sent to BigQuery

def iniciar_ingestao_bigquery():
    """
    Itera sobre as pastas e envia os dados particionados de cada tabela para o BigQuery.
    """
    print("\n=== Início da Ingestão para BigQuery ===")  # Início do processo de ingestão / Starting the ingestion process
    for tabela, pasta in TABELAS_PARQUET.items():  # Itera sobre as tabelas e pastas / Iterates over the tables and folders
        print(f"\nIniciando ingestão para: {tabela}")  # Exibe qual tabela está sendo processada / Displays which table is being processed
        enviar_parquet_para_bigquery(tabela, pasta)  # Envia os dados para BigQuery / Sends data to BigQuery
    print("\nIngestão finalizada com sucesso!")  # Finaliza a ingestão / Finishes the ingestion process

if __name__ == "__main__":
    iniciar_ingestao_bigquery()  # Inicia a ingestão para o BigQuery / Starts the ingestion process for BigQuery
