import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import sagemaker
from sagemaker.sklearn import SKLearn
import os
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# Configuración inicial
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)

# ============================
# 1. FUNCIONES DE PREPROCESAMIENTO
# ============================

def safe_eval(val):
    """Evalúa de forma segura strings que representan listas"""
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(val) if isinstance(val, str) else val
    except:
        return []

def clean_entities(entidades, palabras_excluir):
    """Limpia la lista de entidades"""
    if not isinstance(entidades, list):
        return []
    return [
        e for e in entidades 
        if isinstance(e, str) and not e.strip().isdigit() and e.lower().strip() not in palabras_excluir
    ]

# ============================
# 2. GENERACIÓN DE EDA CORREGIDA
# ============================

def generate_eda(df):
    """Genera análisis exploratorio de datos con manejo robusto de tipos"""
    try:
        # Top frases clave - con manejo de tipos
        all_phrases = []
        for item in df['key_phrases_text']:
            if isinstance(item, list):
                all_phrases.extend(item)
            elif isinstance(item, str):
                try:
                    parsed = ast.literal_eval(item)
                    if isinstance(parsed, list):
                        all_phrases.extend(parsed)
                except:
                    continue
        
        if all_phrases:
            phrases_series = pd.Series(all_phrases).value_counts().head(20)
            plt.figure(figsize=(12, 6))
            sns.barplot(y=phrases_series.index, x=phrases_series.values)
            plt.title("Top 20 Frases Clave")
            plt.show()
        else:
            print("No se encontraron frases clave válidas")

        # Top entidades - con manejo de tipos
        all_entities = []
        for item in df['entities_text']:
            if isinstance(item, list):
                all_entities.extend(item)
            elif isinstance(item, str):
                try:
                    parsed = ast.literal_eval(item)
                    if isinstance(parsed, list):
                        all_entities.extend(parsed)
                except:
                    continue
        
        if all_entities:
            entities_series = pd.Series(all_entities).value_counts().head(20)
            plt.figure(figsize=(12, 6))
            sns.barplot(y=entities_series.index, x=entities_series.values)
            plt.title("Top 20 Entidades")
            plt.show()
        else:
            print("No se encontraron entidades válidas")

    except Exception as e:
        print(f"Error en generate_eda: {str(e)}")
        raise

# ============================
# 3. ENTRENAMIENTO EN SAGEMAKER (VERSIÓN FINAL)
# ============================

def train_sagemaker_model(X_train, y_train):
    """Configura y ejecuta el entrenamiento en SageMaker"""
    role = "arn:aws:iam::802406433795:role/service-role/AmazonSageMaker-ExecutionRole-20250627T194742"
    session = sagemaker.Session()
    
    # Preparar datos
    os.makedirs("./sagemaker_data", exist_ok=True)
    train_data = pd.DataFrame(X_train.toarray())
    train_data['target'] = y_train.values
    train_data = train_data.replace([np.inf, -np.inf], np.nan).dropna()
    train_data.to_csv("./sagemaker_data/train.csv", index=False)
    
    # Subir a S3
    input_train = session.upload_data(
        "./sagemaker_data/train.csv", 
        bucket=session.default_bucket(), 
        key_prefix="secop_model_data"
    )
    
    # Script de entrenamiento con encoding UTF-8 explícito
    script_path = "train_regressor.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write('''# -*- coding: utf-8 -*-
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

if __name__ == "__main__":
    # Cargar datos
    df = pd.read_csv("/opt/ml/input/data/train/train.csv")
    
    # Verificar datos
    if len(df) == 0:
        raise ValueError("No hay datos de entrenamiento")
    
    # Preparar datos
    X = df.drop("target", axis=1)
    y = df["target"].values.ravel()
    
    # Validación cruzada
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print("R2 scores de validación cruzada:", scores)
    print("R2 promedio:", scores.mean())
    
    # Entrenar modelo final
    model.fit(X, y)
    joblib.dump(model, "/opt/ml/model/model.joblib")
''')

    # Configurar y ejecutar estimador
    sklearn_estimator = SKLearn(
        entry_point=script_path,
        role=role,
        instance_type="ml.m5.large",
        framework_version="0.23-1",
        sagemaker_session=session,
        hyperparameters={},
        base_job_name="secop-valor-contrato",
        output_path=f"s3://{session.default_bucket()}/secop_output"
    )
    
    sklearn_estimator.fit({"train": input_train})
    return sklearn_estimator

# ============================
# 4. FLUJO PRINCIPAL
# ============================

def main():
    try:
        # 1. Cargar datos
        bucket = 'secop-public-procurement-dataset'
        df_secop_nlp = pd.read_csv(f's3://{bucket}/secop_nlp_analysis.csv', low_memory=False)
        df_secop_data = pd.read_csv(f's3://{bucket}/secop_data_cleaned.csv', low_memory=False)
        
        # 2. Unir y procesar datos
        df = pd.merge(
            df_secop_data,
            df_secop_nlp[['numero_de_proceso', 'entities_text', 'key_phrases_text']],
            on='numero_de_proceso',
            how='left'
        )
        
        # 3. Limpiar datos de texto
        palabras_excluir = {'bogotá', 'cundinamarca', 'colombia', 'de', 'municipio', 'nacional'}
        df['entities_text'] = df['entities_text'].apply(safe_eval)
        df['key_phrases_text'] = df['key_phrases_text'].apply(safe_eval)
        df['entities_text'] = df['entities_text'].apply(lambda x: clean_entities(x, palabras_excluir))
        df['entities_text_joined'] = df['entities_text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        
        # 4. Preparar datos para modelado
        vectorizer = CountVectorizer(max_features=100, min_df=2)
        X = vectorizer.fit_transform(df['entities_text_joined'].fillna(''))
        y = df['valor_contrato'].copy()
        valid_rows = ~y.isna()
        X = X[valid_rows.values, :]
        y = y[valid_rows]
        
        # 5. Entrenar modelo
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Datos listos. Forma de X_train:", X_train.shape)
        
        estimator = train_sagemaker_model(X_train, y_train)
        print("Entrenamiento completado exitosamente!")
        
        # 6. Generar EDA
        generate_eda(df)
        
    except Exception as e:
        print(f"Error en el proceso principal: {str(e)}")
        raise

if __name__ == "__main__":
    main()