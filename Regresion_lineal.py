import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import glob

# Ruta de la carpeta donde están los archivos CSV
ruta_csv = r'C:\Users\Wadel\OneDrive\Documentos\regresion lineal\*.csv'

# Listar todos los archivos CSV en la ruta especificada
csv_files = glob.glob(ruta_csv)

# Inicializar una lista vacía para almacenar los DataFrames
dfs = []

# Leer cada archivo CSV y agregarlo a la lista
for file in csv_files:
    try:
        df = pd.read_csv(file)
        print(f"Archivo cargado exitosamente: {file}")  # Confirmar archivos leídos
        dfs.append(df)
    except Exception as e:
        print(f"Error al leer el archivo {file}: {e}")

# Combinar todos los DataFrames en uno solo
if dfs:  # Verificar que haya DataFrames en la lista
    combined_df = pd.concat(dfs, ignore_index=True)
    print("Todos los archivos CSV han sido combinados correctamente.")
    print(combined_df.head())  # Mostrar las primeras filas del DataFrame combinado
else:
    print("No se encontraron archivos CSV o no pudieron ser leídos.")

df.head()
df.columns
df.describe()
df.info()

# convierte las columnas seleccionadas al tipo categoría.
df[['Entity','Code']] = df[['Entity','Code']].astype('category')

df.isnull().sum()

# Reemplazar valores faltantes en columnas numéricas con la media
numeric_columns = ['Code', 'Year']  # Lista de columnas numéricas donde se reemplazarán los valores nulos
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean(numeric_only=True))
# Reemplazar valores faltantes en columnas categóricas con la moda
categorical_columns = [
    'Entity',
    'Combined gross enrolment ratio for tertiary education, female',
    'Combined gross enrolment ratio for tertiary education, male',
    'Combined total net enrolment rate, secondary, male',
    'Combined total net enrolment rate, secondary, female',
    'Combined total net enrolment rate, primary, female',
    'Combined total net enrolment rate, primary, male'
]

for column in categorical_columns:  
    df[column] = df[column].fillna(df[column].mode()[0])


df['Code'] = pd.to_numeric(df['Code'], errors='coerce')


print(df['Code'].dtype)


df['Code'].fillna(df['Code'].mean(), inplace=True)

df.drop(columns=['Code'], inplace=True)

df.isnull().sum()

# 1. Histograma
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='Year', bins=10, kde=True)
plt.title('Distribution of Years')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.show()

# Filtramos las columnas necesarias
df_regresion = combined_df[['Year', 'Combined gross enrolment ratio for tertiary education, female']]

# Eliminar filas con valores nulos en estas columnas
df_regresion = df_regresion.dropna()

# Definir la variable dependiente (Y) y las variables independientes (X)
X = df_regresion[['Year']]  # Variable independiente (año)
y = df_regresion['Combined gross enrolment ratio for tertiary education, female']  # Variable dependiente

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# Hacer predicciones con el conjunto de prueba
y_pred = modelo.predict(X_test)

# Evaluar el modelo: Calcular el R² y el MSE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² (coeficiente de determinación): {r2}")
print(f"Error cuadrático medio (MSE): {mse}")

# Graficar los resultados
plt.figure(figsize=(8, 6))

# Gráfico de dispersión de los datos reales vs la predicción
plt.scatter(X_test, y_test, color='blue', label='Datos reales')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicción')

plt.title('Regresión Lineal: Predicción vs Real')
plt.xlabel('Año')
plt.ylabel('Proporción de inscripción en educación terciaria, femenina')
plt.legend()

plt.show()

#####################################################################################33

# Filtramos las columnas necesarias
df_regresion = combined_df[['Year', 'Combined gross enrolment ratio for tertiary education, female']]

# Eliminar filas con valores nulos en estas columnas
df_regresion = df_regresion.dropna()

# Definir la variable dependiente (Y) y las variables independientes (X)
X = df_regresion[['Year']]  # Variable independiente (año)
y = df_regresion['Combined gross enrolment ratio for tertiary education, female']  # Variable dependiente

# Escalar las características para mejorar el rendimiento de la red neuronal
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear la red neuronal
modelo_nn = Sequential()

# Añadir la capa de entrada y la primera capa oculta (neurona ReLU)
modelo_nn.add(Dense(units=64, activation='relu', input_dim=1))

# Añadir una capa oculta adicional
modelo_nn.add(Dense(units=32, activation='relu'))

# Capa de salida (lineal para regresión)
modelo_nn.add(Dense(units=1, activation='linear'))

# Compilar el modelo con optimizador Adam y MSE como función de pérdida
modelo_nn.compile(optimizer=Adam(), loss='mean_squared_error')

# Entrenar el modelo
modelo_nn.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# Hacer predicciones con el conjunto de prueba
y_pred_nn = modelo_nn.predict(X_test)

# Evaluar el modelo: Calcular el R² y el MSE
r2_nn = r2_score(y_test, y_pred_nn)
mse_nn = mean_squared_error(y_test, y_pred_nn)

print(f"R² (coeficiente de determinación): {r2_nn}")
print(f"Error cuadrático medio (MSE): {mse_nn}")

# Graficar los resultados
plt.figure(figsize=(8, 6))

# Gráfico de dispersión de los datos reales vs la predicción
plt.scatter(X_test, y_test, color='blue', label='Datos reales')
plt.plot(X_test, y_pred_nn, color='red', linewidth=2, label='Predicción (Red Neuronal)')

plt.title('Red Neuronal Artificial: Predicción vs Real')
plt.xlabel('Año')
plt.ylabel('Proporción de inscripción en educación terciaria, femenina')
plt.legend()

plt.show()