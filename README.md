# Predicción del Precio de Casas en Ames, Iowa

Este repositorio contiene una solución al reto de Kaggle **"Home Data for ML Course"**, que tiene como objetivo predecir el precio de venta de casas en Ames, Iowa, basado en diversas características de las propiedades.

## Enunciado

### Descripción del reto

El desafío consiste en predecir el precio de venta de cada casa en un conjunto de datos que contiene información sobre 79 variables explicativas que describen casi todos los aspectos de las casas residenciales en Ames, Iowa. Estas variables incluyen características como el número de habitaciones, el tipo de construcción, las condiciones del vecindario, y mucho más.

Con este conjunto de datos, el objetivo es construir un modelo que sea capaz de predecir el precio de venta de una casa en función de sus características.

### Evaluación

La métrica utilizada para evaluar las predicciones es el **RMSE (Root Mean Squared Error)** entre el logaritmo del precio de venta real y el predicho. Esto se hace para asegurarse de que el modelo se enfoque en predecir con precisión tanto las casas caras como las más baratas, equilibrando la importancia de los errores en ambos casos.

### Objetivo

El objetivo de este ejercicio es predecir el precio de venta de las casas utilizando las características de cada propiedad en los conjuntos de datos proporcionados. Las predicciones se realizarán sobre el conjunto de prueba y se enviarán en formato CSV con el formato siguiente:

```csv
Id,SalePrice
1461,169000.1
1462,187724.1233
1463,175221
...
