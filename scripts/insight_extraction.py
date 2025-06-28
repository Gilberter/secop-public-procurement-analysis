import boto3
import pandas as pd

# Configurar cliente de Comprehend
comprehend = boto3.client('comprehend', region_name='us-east-2')

# Leer dataset desde S3
df = pd.read_csv('s3://secop-public-procurement-dataset/secop_data_cleaned.csv')

def analyze_document(text, language='es'):
    if not text or pd.isna(text):
        return None
    
    text = str(text)[:5000]
    
    try:
        entities = comprehend.detect_entities(Text=text, LanguageCode=language)
        key_phrases = comprehend.detect_key_phrases(Text=text, LanguageCode=language)
        sentiment = comprehend.detect_sentiment(Text=text, LanguageCode=language)
        
        return {
            'entities': entities['Entities'],
            'key_phrases': key_phrases['KeyPhrases'],
            'sentiment': sentiment['Sentiment'],
            'sentiment_scores': sentiment['SentimentScore']
        }
    except Exception as e:
        print(f"Error analizando texto: {e}")
        return None

# Mantener índice para asociar con numero_de_proceso
df_valid = df.dropna(subset=['objeto_a_contratar'])
sample_df = df_valid.sample(1000)

# Aplicar análisis
analysis_results = sample_df['objeto_a_contratar'].apply(analyze_document)

# Normalizar resultados
df_nlp = analysis_results.dropna().apply(pd.Series)

# Extraer valores útiles
df_nlp['entities_text'] = df_nlp['entities'].apply(lambda ents: [e['Text'] for e in ents])
df_nlp['key_phrases_text'] = df_nlp['key_phrases'].apply(lambda phrases: [p['Text'] for p in phrases])
df_nlp['sentiment'] = df_nlp['sentiment']
df_nlp[['positive_score', 'negative_score', 'neutral_score', 'mixed_score']] = df_nlp['sentiment_scores'].apply(pd.Series)

# Agregar número de proceso (y opcionalmente objeto_a_contratar si quieres)
df_nlp['numero_de_proceso'] = sample_df.loc[df_nlp.index, 'numero_de_proceso'].values
df_nlp['objeto_a_contratar'] = sample_df.loc[df_nlp.index, 'objeto_a_contratar'].values

# Guardar en S3
s3_client = boto3.client('s3')
bucket_name = 'secop-public-procurement-dataset'
s3_client.put_object(
    Bucket=bucket_name,
    Key='secop_nlp_analysis.csv',
    Body=df_nlp.to_csv(index=False)
)

print("Análisis de texto completado y guardado en S3 y archivo local.")
print(df_nlp[['numero_de_proceso', 'sentiment', 'entities_text', 'key_phrases_text']].head())
