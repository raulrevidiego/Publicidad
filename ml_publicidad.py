# -*- coding: utf-8 -*-
"""ML publicidad"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# -----------------------------
# Carga de datos de publicidad
# -----------------------------
data = pd.read_csv(
    'https://raw.githubusercontent.com/Adrian-Cancino/Machine-Learning-ScikitLearn/refs/heads/main/Conjuntos%20de%20datos/Advertising.csv'
)
# Eliminamos la primera columna (ID)
data = data.iloc[:, 1:]

# Información básica
print(data.info())
print(data.describe())
print("Columnas:", data.columns)

# -----------------------------
# Gráficas de correlación
# -----------------------------
columnas = ['TV', 'Radio', 'Newspaper']  # Variables independientes

for columna in columnas:
    plt.scatter(data[columna], data['Sales'], color='red')
    plt.title(f'Ventas respecto a la publicidad en {columna}')
    plt.xlabel(columna)
    plt.ylabel('Sales')
    plt.show()

# -----------------------------
# Regresión lineal simple con TV
# -----------------------------
x = data['TV'].values.reshape(-1, 1)
y = data['Sales'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

predict = lin_reg.predict(x_test)
print(f"Predicciones:\n{predict}\nReales:\n{y_test}")

mse = mean_squared_error(y_test, predict)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")
print("R2:", r2_score(y_test, predict))

plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, predict, color='blue')
plt.title("Regresión Lineal Simple - TV vs Sales")
plt.xlabel("TV")
plt.ylabel("Sales")
plt.show()

# -----------------------------
# Función para regresión lineal con cualquier variable
# -----------------------------
def modelo_regresion_lineal(independiente):
    x = data[independiente].values.reshape(-1, 1)
    y = data['Sales'].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    predict = lin_reg.predict(x_test)

    print(f"\nVariable: {independiente}")
    print(f"Predicciones:\n{predict}\nReales:\n{y_test}")

    mse = mean_squared_error(y_test, predict)
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse}")
    print("R2:", r2_score(y_test, predict))

    plt.scatter(x_test, y_test, color='red')
    plt.plot(x_test, predict, color='blue')
    plt.title(f"Regresión Lineal - {independiente} vs Sales")
    plt.xlabel(independiente)
    plt.ylabel("Sales")
    plt.show()


modelo_regresion_lineal('Radio')
modelo_regresion_lineal('Newspaper')

# -----------------------------
# Regresión múltiple (sin Radio)
# -----------------------------
x = data.drop(['Radio', 'Sales'], axis=1).values
y = data['Sales'].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
predict = lin_reg.predict(x_test)

print("\nRegresión múltiple (TV + Newspaper)")
print(f"Predicciones:\n{predict}\nReales:\n{y_test}")
print("RMSE:", np.sqrt(mean_squared_error(y_test, predict)))
print("R2:", r2_score(y_test, predict))

sns.regplot(x=y_test, y=predict)
plt.title("Regresión múltiple (TV + Newspaper)")
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.show()

# -----------------------------
# Regresión múltiple (sin Newspaper)
# -----------------------------
x = data.drop(['Newspaper', 'Sales'], axis=1).values
y = data['Sales'].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
predict = lin_reg.predict(x_test)

print("\nRegresión múltiple (TV + Radio)")
print(f"Predicciones:\n{predict}\nReales:\n{y_test}")
print("RMSE:", np.sqrt(mean_squared_error(y_test, predict)))
print("R2:", r2_score(y_test, predict))

sns.regplot(x=y_test, y=predict)
plt.title("Regresión múltiple (TV + Radio)")
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.show()

# -----------------------------
# Regresión polinómica (ejemplo salario)
# -----------------------------
years = list(range(1, 11))
position = [f"Puesto {i}" for i in range(1, 11)]
salary = [1000.0, 1150.0, 1260.0, 1600.0, 1900.0, 2400.0, 3000.0, 4000.0, 5200.0, 6000.0]

df_salary = pd.DataFrame({"posición": position, "años": years, "salario": salary})

plt.scatter(df_salary["posición"], df_salary["salario"])
plt.title("Salario por Puesto")
plt.xlabel("Puesto")
plt.ylabel("Salario")
plt.show()

x = df_salary["años"].values.reshape(-1, 1)
y = df_salary["salario"].values

# Regresión lineal simple
regresion_lineal = LinearRegression()
regresion_lineal.fit(x, y)
plt.scatter(x, y, color='blue')
plt.plot(x, regresion_lineal.predict(x), color='black')
plt.title("Regresión Lineal - Salario vs Años")
plt.xlabel("Años")
plt.ylabel("Salario")
plt.show()

# Regresión polinómica (grado 9)
poly = PolynomialFeatures(degree=9)
x_poly = poly.fit_transform(x)
regresion_poly = LinearRegression()
regresion_poly.fit(x_poly, y)

plt.scatter(x, y)
plt.plot(x, regresion_poly.predict(x_poly), color='red')
plt.title("Regresión Polinómica - Salario vs Años")
plt.xlabel("Años")
plt.ylabel("Salario")
plt.show()
