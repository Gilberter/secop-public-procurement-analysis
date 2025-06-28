import pandas as pd
import awswrangler as wr
import boto3
import unicodedata


df = wr.s3.read_csv('s3://secop-public-procurement-dataset/secop_data.csv')

print(f"DataFrame data types:\n{df.dtypes}")

# Change column names

df.rename(columns={
    'nivel_entidad': 'nivel_entidad',
    'codigo_entidad_en_secop': 'codigo_entidad_en_secop',
    'nombre_de_la_entidad': 'nombre_de_la_entidad',
    'nit_de_la_entidad': 'nit_de_la_entidad',
    'departamento_entidad': 'departamento_entidad',
    'municipio_entidad': 'municipio_entidad',
    'estado_del_proceso': 'estado_del_proceso',
    'modalidad_de_contrataci_n': 'modalidad_de_contratacion',
    'objeto_a_contratar': 'objeto_a_contratar',
    'objeto_del_proceso': 'objeto_del_proceso',
    'tipo_de_contrato': 'tipo_de_contrato',
    'fecha_de_firma_del_contrato': 'fecha_de_firma_del_contrato',
    'fecha_inicio_ejecuci_n': 'fecha_inicio_ejecucion',
    'fecha_fin_ejecuci_n': 'fecha_fin_ejecucion',
    'numero_del_contrato': 'numero_del_contrato',
    'numero_de_proceso': 'numero_de_proceso',
    'valor_contrato': 'valor_contrato',
    'nom_raz_social_contratista': 'nom_raz_social_contratista',
    'url_contrato': 'url_contrato',
    'origen': 'origen',
    'tipo_documento_proveedor': 'tipo_documento_proveedor',
    'documento_proveedor': 'documento_proveedor'
}, inplace=True)


# ETL Data Cleaning
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .map(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8'))
        .str.replace(r'\W+', '_', regex=True)
    )
    return df

def parse_dates(df: pd.DataFrame, date_columns: list) -> pd.DataFrame:
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def clean_text_columns(df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    for col in text_columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r'\s+', ' ', regex=True)
        )
    return df

def normalize_categoricals(df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    for col in categorical_columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.lower()
            .map(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8'))
            .str.strip()
        )
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=['numero_de_proceso', 'numero_del_contrato'])

def filter_invalid_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['valor_contrato'] > 0]
    df = df[df['fecha_inicio_ejecucion'] <= df['fecha_fin_ejecucion']]
    return df

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df['duracion_dias'] = (df['fecha_fin_ejecucion'] - df['fecha_inicio_ejecucion']).dt.days
    df['anio_contrato'] = df['fecha_de_firma_del_contrato'].dt.year
    return df

def convert_column_types(df: pd.DataFrame) -> pd.DataFrame:
    df['codigo_entidad_en_secop'] = df['codigo_entidad_en_secop'].astype(str)
    df['documento_proveedor'] = df['documento_proveedor'].astype(str)
    return df

def clean_secop_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Clean and standardize column names
    df = clean_column_names(df)

    # Convert dates
    date_columns = [
        'fecha_de_firma_del_contrato',
        'fecha_inicio_ejecucion',
        'fecha_fin_ejecucion'
    ]
    df = parse_dates(df, date_columns)

    # Clean text
    text_columns = [
        'objeto_a_contratar',
        'objeto_del_proceso',
        'nom_raz_social_contratista',
        'nombre_de_la_entidad',
        'municipio_entidad'
    ]
    df = clean_text_columns(df, text_columns)

    # Normalize categoricals
    categorical_columns = [
        'modalidad_de_contratacion',
        'estado_del_proceso',
        'tipo_de_contrato',
        'departamento_entidad'
    ]
    df = normalize_categoricals(df, categorical_columns)

    # Remove duplicates
    df = remove_duplicates(df)

    # Filter invalid
    df = filter_invalid_data(df)

    # Derive additional columns
    df = add_derived_columns(df)

    # Fix column types
    df = convert_column_types(df)

    return df

# Clean the DataFrame
df_cleaned = clean_secop_dataframe(df)
# Save the cleaned DataFrame back to S3
s3_client = boto3.client('s3')
bucket_name = 'secop-public-procurement-dataset'
s3_client.put_object(
    Bucket=bucket_name,
    Key='secop_data_cleaned.csv',
    Body=df_cleaned.to_csv(index=False)
)
# Display the cleaned DataFrame
print(f"Cleaned DataFrame data types:\n{df_cleaned.dtypes}")
# Display the first few rows of the cleaned DataFrame
