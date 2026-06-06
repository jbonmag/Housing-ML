# Housing ML

Modelo de machine learning para predecir precios de vivienda en Ames, Iowa, basado en el reto de Kaggle Home Data for ML Course. El repositorio incluye los CSV de entrenamiento y prueba, un notebook exploratorio y un script reproducible para entrenar el modelo y generar `submission.csv`.

## Objetivo

Predecir `SalePrice` a partir de variables numéricas y categóricas de cada propiedad. La métrica del reto es RMSE sobre el logaritmo del precio, equivalente a RMSLE sobre el precio original.

## Estructura

```text
.
├── train.csv
├── test.csv
├── sample_submission.csv
├── data_description.txt
├── modelo_predictivo.ipynb
├── src/
│   └── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Enfoque

El script principal usa un pipeline de scikit-learn:

1. Separación de variables predictoras y objetivo.
2. Transformación `log1p` de `SalePrice`.
3. Imputación de valores nulos.
4. Codificación one-hot de variables categóricas.
5. Entrenamiento con `RandomForestRegressor`.
6. Validación cruzada con RMSLE.
7. Generación de `submission.csv`.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En Linux o macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

Entrenar y generar el archivo de envío:

```bash
python src/train_model.py
```

Usar rutas personalizadas:

```bash
python src/train_model.py --train train.csv --test test.csv --output outputs/submission.csv
```

## Resultado

La ejecución imprime las puntuaciones de validación cruzada y crea un CSV con el formato esperado por Kaggle:

```csv
Id,SalePrice
1461,169000.1
1462,187724.1
```

## Mejoras posibles

- Ajuste de hiperparámetros con `RandomizedSearchCV`.
- Comparación con modelos de boosting.
- Análisis de importancia de variables.
- Validación de outliers y transformación de variables sesgadas.

