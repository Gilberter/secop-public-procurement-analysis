# -*- coding: utf-8 -*-
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

if __name__ == "__main__":
    # Cargar datos
    df = pd.read_csv("/opt/ml/input/data/train/train.csv")
    
    # Preparar datos
    X = df.drop("target", axis=1)
    y = df["target"].values.ravel()
    
    # Validación cruzada
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=3, scoring='r2')
    print("R2 scores:", scores)
    print("R2 promedio:", scores.mean())
    
    # Entrenar modelo final
    model.fit(X, y)
    joblib.dump(model, "/opt/ml/model/model.joblib")
