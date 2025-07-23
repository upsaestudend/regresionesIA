import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Título
st.title("Predicción de Nota Final con Regresión Lineal Múltiple")

# Subir archivo CSV
st.sidebar.header("1. Cargar archivo CSV")
archivo = st.sidebar.file_uploader("Sube tu archivo de datos", type=["csv"])

if archivo:
    df = pd.read_csv(archivo)
else:
    # Si no se sube archivo, usar archivo por defecto generado
    df = pd.read_csv("evaluaciones_estudiantes_1000.csv")
    st.info("Usando archivo por defecto: evaluaciones_estudiantes_1000.csv")

# Vista previa
st.subheader("Vista previa de los datos")
st.write(df.head())

# Visualización: Pairplot
st.subheader("Relaciones entre variables")
fig1 = sns.pairplot(df)
st.pyplot(fig1)

# Matriz de correlación
st.subheader("Matriz de correlación")
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# Variables para el modelo
X = df[['Parcial1', 'Parcial2', 'Proyecto', 'Examen_Final']]
y = df['Nota_Final']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# Evaluación
st.subheader("Evaluación del Modelo")
st.write("Error Cuadrático Medio (MSE):", round(mean_squared_error(y_test, y_pred), 2))
st.write("Coeficiente de Determinación (R²):", round(r2_score(y_test, y_pred), 2))

# Comparación gráfica
st.subheader("Comparación: Notas Reales vs Predichas")
fig3, ax3 = plt.subplots()
ax3.scatter(y_test, y_pred, alpha=0.5)
ax3.plot([0, 100], [0, 100], 'r--')
ax3.set_xlabel("Nota Real")
ax3.set_ylabel("Nota Predicha")
ax3.set_title("Comparación de Notas")
st.pyplot(fig3)

# Formulario de predicción
st.sidebar.header("2. Predecir nueva nota")
p1 = st.sidebar.slider("Parcial 1 (0-20)", 0, 20, 14)
p2 = st.sidebar.slider("Parcial 2 (0-20)", 0, 20, 15)
proy = st.sidebar.slider("Proyecto (0-20)", 0, 20, 16)
exam = st.sidebar.slider("Examen Final (0-40)", 0, 40, 30)

if st.sidebar.button("Predecir Nota Final"):
    nueva_pred = modelo.predict([[p1, p2, proy, exam]])
    st.sidebar.success(f"Nota Final estimada: {nueva_pred[0]:.2f}")