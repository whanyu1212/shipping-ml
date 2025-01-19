import pandas as pd
from loguru import logger  # type: ignore
from google.cloud import bigquery
from google.oauth2 import service_account  # type: ignore


def write_to_bigquery(df: pd.DataFrame, table_id: str, secret_key_path: str):
    credentials = service_account.Credentials.from_service_account_file(
        secret_key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    client = bigquery.Client(
        credentials=credentials,
        project=credentials.project_id,
    )

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
    )
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)

    job.result()

    table = client.get_table(table_id)
    logger.info(
        "Loaded {} rows and {} columns to {}".format(
            table.num_rows, len(table.schema), table_id
        )
    )


# Sample usage
if __name__ == "__main__":
    df = pd.read_csv("./data/rent_apartments.csv")

    write_to_bigquery(
        df,
        "fleet-anagram-244304.ml_datasets.rent_apartments",
        "./fleet-anagram-244304-21f29c9fbd10.json",
    )
