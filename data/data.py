import pandas as pd
import sodapy
import boto3

client = sodapy.Socrata("www.datos.gov.co", None)

results = client.get("rpmr-utcd", limit=100000)

results_df = pd.DataFrame.from_records(results)

s3_client = boto3.client('s3')
bucket_name = 'secop-public-procurement-dataset'
s3_client.put_object(
    Bucket=bucket_name,
    Key='secop_data.csv',
    Body=results_df.to_csv(index=False)
)
